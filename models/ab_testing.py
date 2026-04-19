"""
A/B Testing framework for model variants.
Enables controlled experiments with different model configurations.
"""
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, Session, mapped_column

from db.models import Base
from utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentStatus(Enum):
    """Experiment status enumeration."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class VariantAllocation(Enum):
    """Variant allocation strategy."""

    RANDOM = "random"
    HASH_BASED = "hash_based"
    WEIGHTED = "weighted"


@dataclass
class ModelVariant:
    """Model variant configuration."""

    variant_id: str
    model_type: str
    hyperparameters: dict[str, Any]
    feature_config: dict[str, Any]
    weight: float = 1.0


@dataclass
class ExperimentMetrics:
    """Experiment metrics for comparison."""

    variant_id: str
    predictions_count: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    win_rate: float
    avg_pnl: float
    max_drawdown: float


class ABExperiment(Base):
    """A/B experiment model."""

    __tablename__ = "ab_experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(Text)
    symbol: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), default="draft")
    allocation_strategy: Mapped[str] = mapped_column(String(32), default="hash_based")
    variants_config: Mapped[dict] = mapped_column(JSON, default=dict)
    start_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    control_variant_id: Mapped[str] = mapped_column(String(128))
    winner_variant_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ABPrediction(Base):
    """A/B prediction tracking."""

    __tablename__ = "ab_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(128), index=True)
    variant_id: Mapped[str] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(128), index=True)
    session_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    pred_close: Mapped[float] = mapped_column(Float)
    pred_direction: Mapped[str] = mapped_column(String(8))
    confidence: Mapped[float] = mapped_column(Float)
    actual_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_direction: Mapped[str | None] = mapped_column(String(8), nullable=True)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ABTestingFramework:
    """A/B testing framework for model experiments."""

    def __init__(self, db: Session):
        self.db = db

    def create_experiment(
        self,
        name: str,
        symbol: str,
        variants: list[ModelVariant],
        control_variant_id: str,
        description: str = "",
        allocation_strategy: VariantAllocation = VariantAllocation.HASH_BASED,
    ) -> str:
        """
        Create a new A/B experiment.

        Args:
            name: Experiment name
            symbol: Trading symbol
            variants: List of model variants to test
            control_variant_id: ID of the control variant
            description: Experiment description
            allocation_strategy: How to allocate traffic to variants

        Returns:
            experiment_id
        """
        experiment_id = self._generate_experiment_id(name, symbol)

        variants_config = {
            "variants": [
                {
                    "variant_id": v.variant_id,
                    "model_type": v.model_type,
                    "hyperparameters": v.hyperparameters,
                    "feature_config": v.feature_config,
                    "weight": v.weight,
                }
                for v in variants
            ]
        }

        experiment = ABExperiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            symbol=symbol,
            status=ExperimentStatus.DRAFT.value,
            allocation_strategy=allocation_strategy.value,
            variants_config=variants_config,
            control_variant_id=control_variant_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        self.db.add(experiment)
        self.db.commit()

        logger.info(f"Created experiment {experiment_id} with {len(variants)} variants")
        return experiment_id

    def start_experiment(self, experiment_id: str) -> None:
        """Start an experiment."""
        experiment = self._get_experiment(experiment_id)
        experiment.status = ExperimentStatus.RUNNING.value
        experiment.start_date = datetime.now()
        experiment.updated_at = datetime.now()
        self.db.commit()
        logger.info(f"Started experiment {experiment_id}")

    def stop_experiment(self, experiment_id: str) -> None:
        """Stop an experiment."""
        experiment = self._get_experiment(experiment_id)
        experiment.status = ExperimentStatus.COMPLETED.value
        experiment.end_date = datetime.now()
        experiment.updated_at = datetime.now()
        self.db.commit()
        logger.info(f"Stopped experiment {experiment_id}")

    def get_variant_for_prediction(self, experiment_id: str, symbol: str, date: datetime) -> str:
        """
        Get variant assignment for a prediction.

        Args:
            experiment_id: Experiment ID
            symbol: Trading symbol
            date: Prediction date

        Returns:
            variant_id
        """
        experiment = self._get_experiment(experiment_id)

        if experiment.status != ExperimentStatus.RUNNING.value:
            return experiment.control_variant_id

        variants = experiment.variants_config.get("variants", [])

        if experiment.allocation_strategy == VariantAllocation.HASH_BASED.value:
            return self._hash_based_allocation(symbol, date, variants)
        elif experiment.allocation_strategy == VariantAllocation.WEIGHTED.value:
            return self._weighted_allocation(variants)
        else:
            return self._random_allocation(variants)

    def record_prediction(
        self,
        experiment_id: str,
        variant_id: str,
        symbol: str,
        session_date: datetime,
        pred_close: float,
        pred_direction: str,
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a prediction for A/B testing."""
        prediction = ABPrediction(
            experiment_id=experiment_id,
            variant_id=variant_id,
            symbol=symbol,
            session_date=session_date,
            pred_close=pred_close,
            pred_direction=pred_direction,
            confidence=confidence,
            metadata_json=metadata or {},
            created_at=datetime.now(),
        )

        self.db.add(prediction)
        self.db.commit()

    def update_prediction_outcome(
        self,
        experiment_id: str,
        symbol: str,
        session_date: datetime,
        actual_close: float,
        actual_direction: str,
        pnl: float,
    ) -> None:
        """Update prediction with actual outcome."""
        from sqlalchemy import and_, select

        predictions = (
            self.db.execute(
                select(ABPrediction).where(
                    and_(
                        ABPrediction.experiment_id == experiment_id,
                        ABPrediction.symbol == symbol,
                        ABPrediction.session_date == session_date,
                    )
                )
            )
            .scalars()
            .all()
        )

        for pred in predictions:
            pred.actual_close = actual_close
            pred.actual_direction = actual_direction
            pred.is_correct = pred.pred_direction == actual_direction
            pred.pnl = pnl

        self.db.commit()

    def get_experiment_metrics(self, experiment_id: str) -> list[ExperimentMetrics]:
        """Get metrics for all variants in an experiment."""
        from sqlalchemy import and_, func, select

        experiment = self._get_experiment(experiment_id)
        variants = experiment.variants_config.get("variants", [])

        metrics_list = []

        for variant in variants:
            variant_id = variant["variant_id"]

            # Query predictions for this variant
            stats = self.db.execute(
                select(
                    func.count(ABPrediction.id).label("count"),
                    func.avg(ABPrediction.pnl).label("avg_pnl"),
                    func.sum(
                        func.cast(ABPrediction.is_correct, Integer)
                    ).label("correct_count"),
                ).where(
                    and_(
                        ABPrediction.experiment_id == experiment_id,
                        ABPrediction.variant_id == variant_id,
                        ABPrediction.actual_close.isnot(None),
                    )
                )
            ).first()

            if stats and stats.count > 0:
                accuracy = (stats.correct_count or 0) / stats.count
                metrics_list.append(
                    ExperimentMetrics(
                        variant_id=variant_id,
                        predictions_count=stats.count,
                        accuracy=accuracy,
                        precision=0.0,  # Calculate from confusion matrix
                        recall=0.0,
                        f1_score=0.0,
                        sharpe_ratio=0.0,  # Calculate from PnL series
                        win_rate=accuracy,
                        avg_pnl=stats.avg_pnl or 0.0,
                        max_drawdown=0.0,  # Calculate from cumulative PnL
                    )
                )

        return metrics_list

    def declare_winner(self, experiment_id: str, winner_variant_id: str) -> None:
        """Declare a winner for the experiment."""
        experiment = self._get_experiment(experiment_id)
        experiment.winner_variant_id = winner_variant_id
        experiment.status = ExperimentStatus.COMPLETED.value
        experiment.end_date = datetime.now()
        experiment.updated_at = datetime.now()
        self.db.commit()
        logger.info(f"Declared winner for experiment {experiment_id}: {winner_variant_id}")

    def _get_experiment(self, experiment_id: str) -> ABExperiment:
        """Get experiment by ID."""
        from sqlalchemy import select

        experiment = self.db.scalar(
            select(ABExperiment).where(ABExperiment.experiment_id == experiment_id)
        )
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        return experiment

    def _generate_experiment_id(self, name: str, symbol: str) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().isoformat()
        raw = f"{name}_{symbol}_{timestamp}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _hash_based_allocation(
        self, symbol: str, date: datetime, variants: list[dict]
    ) -> str:
        """Hash-based variant allocation for consistent assignment."""
        key = f"{symbol}_{date.date()}"
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        idx = hash_val % len(variants)
        return variants[idx]["variant_id"]

    def _weighted_allocation(self, variants: list[dict]) -> str:
        """Weighted random allocation."""
        import random

        weights = [v.get("weight", 1.0) for v in variants]
        return random.choices([v["variant_id"] for v in variants], weights=weights)[0]

    def _random_allocation(self, variants: list[dict]) -> str:
        """Random allocation."""
        import random

        return random.choice(variants)["variant_id"]

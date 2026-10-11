"""Internal read-only delivery/review evidence; no send, staging or acceptance."""

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from shared_lib.infrastructure.annotation_store import AnnotationIntegrityError
from shared_lib.infrastructure.proposal_readiness_store import (
    DynamoProposalReadinessStore,
)
from shared_lib.models.analysis_availability import ComponentAvailabilityV1
from shared_lib.models.analysis_timeline import content_sha256


@dataclass(frozen=True)
class ComponentResultView:
    component: str
    publication_state: str
    declared_item_count: int | None
    published_at: datetime | None
    review: ComponentAvailabilityV1


@dataclass(frozen=True)
class ResultStatusView:
    run_id: UUID
    observed_at: datetime
    components: tuple[ComponentResultView, ...]


async def read_result_status(run, outbox, *, guard):
    """Read the consumer row BEFORE publication to tolerate concurrent delivery.

    Receipt/readiness pairs share one strongly read row; publication can be newer.
    Missing acknowledgements never negate a real consumer readiness certificate.
    """
    if not callable(guard) or outbox.components.proposals is not outbox.proposals:
        raise ValueError("shared run store and current authority required")
    spec, proposals, components = run.spec, outbox.proposals, outbox.components
    await guard()
    row = await proposals.annotations._get(
        proposals.tables.runs, {"run_id": str(spec.run_id)}
    )
    await guard()
    if row is None or row.get("content_hash") != content_sha256(run):
        raise AnnotationIntegrityError("registered result-status run changed")
    if any(
        key in row
        for key in (
            "result",
            "result_hash",
            "complete_readiness",
            "complete_readiness_hash",
        )
    ):
        raise AnnotationIntegrityError("position DAG requires component result intake")
    readiness = DynamoProposalReadinessStore(proposals)
    views = []
    for lane in ("positions", "techniques"):
        await guard()
        receipt = await components.receipt(
            spec.analysis_id, spec.run_id, lane, actor=spec.owner, _row=row
        )
        await guard()
        marker = await readiness.read(run, lane, receipt, actor=spec.owner, _row=row)
        await guard()
        publication = await outbox.read(spec.run_id, lane)
        await guard()
        # outbox.read verifies the immutable artifact as well as declaration/ack.
        if publication is None:
            if receipt is not None or f"{lane}_publication_message_hash" in row:
                raise AnnotationIntegrityError("consumer evidence without publication")
            state, count, sent_at = "not_prepared", None, None
        else:
            record, sent = publication
            if record.run_sha256 != content_sha256(run):
                raise AnnotationIntegrityError("publication differs from observed run")
            if receipt is not None and (
                receipt.message != record.message
                or receipt.result_sha256 != record.result_sha256
            ):
                raise AnnotationIntegrityError(
                    "consumer evidence differs from publication"
                )
            state = "prepared" if sent is None else "acknowledged"
            count = record.message.payload.item_count
            sent_at = None if sent is None else sent.published_at
        review = ComponentAvailabilityV1(state="not_received")
        if receipt is not None:
            review = ComponentAvailabilityV1(
                state="received" if marker is None else "ready",
                item_count=receipt.message.payload.item_count,
                proposal_count=(
                    None if marker is None else receipt.message.payload.item_count
                ),
                received_at=receipt.ingested_at,
                ready_at=None if marker is None else marker.ready_at,
            )
        views.append(ComponentResultView(lane, state, count, sent_at, review))
    await guard()
    observed = datetime.now(timezone.utc)
    if any(
        stamp is not None and stamp > observed
        for view in views
        for stamp in (view.published_at, view.review.received_at, view.review.ready_at)
    ):
        raise AnnotationIntegrityError("result-status timestamp is in the future")
    return ResultStatusView(spec.run_id, observed, tuple(views))

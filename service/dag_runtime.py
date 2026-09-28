"""Explicit engine composition; no legacy startup or implicit paid recovery."""

import asyncio

from shared_lib.infrastructure.dag_internal_transport import (
    DagRunAuthority,
    scheduler_enqueue_endpoint,
    DagInternalEndpoint,
)
from shared_lib.infrastructure.dag_service_runtime import DagServiceRuntime

from service.analysis_run_schedule import PositionRunSchedule, OwnerPositionRunSchedule
from service.analysis_gemini import qualified_model_versions
from service.result_delivery_supervisor import ResultDeliverySupervisor


def engine_service_runtime(
    dispatcher,
    sns_client,
    *,
    listener_settings,
    enqueue_keys,
    enqueue_audience,
    observe_schedule,
    observe_delivery,
    on_failure,
    interval_seconds=5,
):
    """Supervise real scheduling and publication alongside the signed listener.

    The parent host owns the dispatcher, authority HTTP clients and native clients
    until this runtime AND public requests have drained (use serve_dag_host).
    The dispatcher must already have the explicitly qualified model configuration;
    this factory does not qualify a model, invent a preset, provision infrastructure,
    or enable the legacy automatic-recovery loop.
    """
    qualified_model_versions(dispatcher.qualified_models)
    if not isinstance(dispatcher.api_key, str) or not dispatcher.api_key.strip():
        raise ValueError("explicit Gemini credential required before startup")
    if not isinstance(dispatcher.authority, DagRunAuthority):
        raise ValueError("engine requires signed current account/session authority")
    if not enqueue_keys or {key.owner_kind for key in enqueue_keys} != {
        "account",
        "session",
    }:
        raise ValueError("both owner-bound admission credentials are required")
    access_secrets = {
        client.key.secret for client in dispatcher.authority.clients.values()
    }
    if len(access_secrets) != 2 or any(
        key.secret in access_secrets for key in enqueue_keys
    ):
        raise ValueError("access and enqueue require separate host credentials")
    if len({key.secret for key in enqueue_keys}) != len(enqueue_keys):
        raise ValueError("enqueue credentials cannot share secrets")
    if not callable(observe_schedule) or not callable(observe_delivery):
        raise ValueError("both worker observers are required")

    schedule = PositionRunSchedule(dispatcher)
    delivery = ResultDeliverySupervisor(
        dispatcher.outbox, sns_client, authority=dispatcher.authority
    )

    async def schedule_work(stop):
        await schedule.run(
            stop, observe=observe_schedule, interval_seconds=interval_seconds
        )

    async def delivery_work(stop):
        await delivery.run(
            stop, observe=observe_delivery, interval_seconds=interval_seconds
        )

    return DagServiceRuntime(
        application=scheduler_enqueue_endpoint(
            schedule, audience=enqueue_audience, keys=enqueue_keys
        ),
        settings=listener_settings,
        workers={"schedule": schedule_work, "delivery": delivery_work},
        on_failure=on_failure,
    )


def partitioned_engine_service_runtime(
    dispatchers,
    sns_clients,
    *,
    listener_settings,
    enqueue_keys,
    enqueue_audience,
    observe_schedule,
    observe_delivery,
    on_failure,
    interval_seconds=5,
):
    """Two explicit stores, credential-selected enqueue, one execution permit.

    Parent owns clients and supplies fresh, not-yet-running dispatchers. Observers
    receive (owner_kind, page). Existing single-store composition remains available
    for its isolated harness; this deployment factory never falls back to it.
    """
    owners = {"account", "session"}
    if set(dispatchers) != owners or set(sns_clients) != owners:
        raise ValueError("both explicit owner stores and publication clients required")
    if dispatchers["account"] is dispatchers["session"]:
        raise ValueError("separate owner dispatchers required")
    if (
        dispatchers["account"].proposals.tables.runs
        == dispatchers["session"].proposals.tables.runs
    ):
        raise ValueError("separate owner registration stores required")
    if (
        dispatchers["account"].outbox.table.name
        == dispatchers["session"].outbox.table.name
    ):
        raise ValueError("separate owner publication stores required")
    if not callable(observe_schedule) or not callable(observe_delivery):
        raise ValueError("both owner-aware observers required")
    if not enqueue_keys or {key.owner_kind for key in enqueue_keys} != owners:
        raise ValueError("both owner-bound enqueue credentials required")
    secrets = [key.secret for key in enqueue_keys]
    if len(set(secrets)) != len(secrets):
        raise ValueError("enqueue secrets must be distinct")
    for dispatcher in dispatchers.values():
        qualified_model_versions(dispatcher.qualified_models)
        if not isinstance(dispatcher.api_key, str) or not dispatcher.api_key.strip():
            raise ValueError("explicit Gemini credential required before startup")
        if not isinstance(dispatcher.authority, DagRunAuthority):
            raise ValueError("signed current host authority required")
        access = {client.key.secret for client in dispatcher.authority.clients.values()}
        if len(access) != 2 or access.intersection(secrets):
            raise ValueError("distinct access/enqueue secrets required")

    schedules = {
        owner: OwnerPositionRunSchedule(dispatchers[owner], owner_kind=owner)
        for owner in owners
    }

    async def enqueue(run_id, owner):
        # Signature verification chooses exactly one store. No probe of the other.
        return await schedules[owner].enqueue(run_id)

    endpoint = DagInternalEndpoint(
        action="enqueue", audience=enqueue_audience, keys=enqueue_keys, handler=enqueue
    )
    workers = {}

    def owner_workers(owner):
        dispatcher = dispatchers[owner]

        async def authority(run):
            if run.spec.owner.kind != owner:
                raise PermissionError("publication owner differs from delivery lane")
            await dispatcher.authority(run)

        delivery = ResultDeliverySupervisor(
            dispatcher.outbox, sns_clients[owner], authority=authority
        )

        async def schedule_work(stop):
            async def observe(page):
                await observe_schedule(owner, page)

            await schedules[owner].run(
                stop, observe=observe, interval_seconds=interval_seconds
            )

        async def delivery_work(stop):
            async def observe(page):
                await observe_delivery(owner, page)

            await delivery.run(stop, observe=observe, interval_seconds=interval_seconds)

        return schedule_work, delivery_work

    for owner in sorted(owners):
        workers[f"{owner}_schedule"], workers[f"{owner}_delivery"] = owner_workers(
            owner
        )
    runtime = DagServiceRuntime(
        application=endpoint,
        settings=listener_settings,
        workers=workers,
        on_failure=on_failure,
    )
    shared_slots = asyncio.Semaphore(1)
    for dispatcher in dispatchers.values():
        dispatcher.slots = shared_slots
    return runtime

"""Account-only current DAG runtime composition.

The current account host's binding of ``service.single_lane_dag_server``: every
entry point here refuses anything but the account lane.
"""

from __future__ import annotations

from shared_lib.infrastructure.dag_internal_transport import (
    DagInternalClient,
    DagServiceKey,
)

from service.single_lane_dag_server import (
    SingleLaneDagRunAuthority,
    single_lane_engine_service_runtime,
    single_lane_position_dispatcher,
    single_lane_run_recovery,
)


class AccountOnlyDagRunAuthority(SingleLaneDagRunAuthority):
    """Signed authority containing exactly one genuine account host client."""

    def __init__(self, account: DagInternalClient):
        if not isinstance(account, DagInternalClient):
            raise TypeError("one signed account authority client is required")
        if account.key.owner_kind != "account":
            raise ValueError("account authority client must be account-bound")
        super().__init__(account)


def current_account_authority_client(
    origin, *, access_key: DagServiceKey, audience, local_test=False, transport=None
):
    """Create the one signed account gateway client for an account dispatcher."""
    if not isinstance(access_key, DagServiceKey) or access_key.owner_kind != "account":
        raise ValueError("account-bound gateway access credential required")
    return AccountOnlyDagRunAuthority(
        DagInternalClient(
            origin,
            key=access_key,
            audience=audience,
            local_test=local_test,
            transport=transport,
        )
    )


def current_account_position_dispatcher(*, authority, **resources):
    """Build the one real position dispatcher over admitted current AI stores."""
    if not isinstance(authority, AccountOnlyDagRunAuthority):
        raise ValueError("an explicit current account authority is required")
    return single_lane_position_dispatcher(authority=authority, **resources)


def _require_account(dispatcher):
    if not isinstance(getattr(dispatcher, "authority", None), AccountOnlyDagRunAuthority):
        raise ValueError("current account dispatcher requires account-only signed authority")


def current_account_run_recovery(dispatcher):
    """Owner-requested recovery queue for the account lane of ``dispatcher``."""
    _require_account(dispatcher)
    return single_lane_run_recovery(dispatcher)


def current_account_engine_service_runtime(dispatcher, sns_client, **settings):
    """Build supervised account scheduling, enqueue and result delivery workers.

    See ``single_lane_engine_service_runtime`` for the settings and the
    lifetime contract.
    """
    _require_account(dispatcher)
    return single_lane_engine_service_runtime(dispatcher, sns_client, **settings)

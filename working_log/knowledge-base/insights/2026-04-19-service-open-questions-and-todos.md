---
date: 2026-04-19
category: insight
tags: [service, lifecycle, resume, todo, open-questions]
status: active
---

# Service Open Questions and TODO Reality Snapshot

## Context
Knowledge base entries describe intended service behavior, but current code contains explicit TODOs, questions, and a production-blocker note that indicate parts of the lifecycle are still unresolved.

## Content
Open items below are copied from code comments as-is and should be treated as unresolved:

- `service/reconciler.py`: `TODO(PRODUCTION-BLOCKER): This is currently a stub. The bounce-proof guarantee requires scanning Keyspaces ...`
- `service/routes.py`: `TODO: user id needs to be passed in from the request. update TrackRequest model to include user_id.`
- `service/routes.py`: `TODO: by default resume should use the latest checkpoint. Detection response should only be about detection during track. If the latest checkpoint in track stage is not available, use the latest checkpoint. then dont bother with handing detection response.`
- `service/routes.py`: `TODO: This is not used meaningfully anywhere. Remove it.`
- `service/routes.py`: `TODO: This should be removed without hesitation: job should be canceled before it is resumed.`
- `service/routes.py`: `TODO: is this is a resumed job the stage should be track.`
- `service/worker.py`: `TODO: tracking progress_percent should be decided based on real frames the total frames in the video.`
- `service/worker.py`: `TODO: update job lifecycle and checkpoint here as well.`
- `service/worker.py`: `TODO: suspend is not the expected behavior. Instead, a checkpoint needs to becreated`
- `service/worker.py`: `TODO: start frame should be the global start frame as this is the merged tracking json.`
- `service/worker.py`: `TODO: Give a big fix of the checkpoint data hsere as when the job is cancelled during because of waiting for detection correction, the checkpoint must be updated like a mid-track checkpoint. and the flag completed should be False.`
- `service/worker.py`: `start frame should be the start frame of the new frame waiting to be tracked?`
- `service/worker.py`: `TODO: in parallel the upscale and analysis.`
- `service/worker.py`: `TODO: annotate the video in a coroutine and dont block the upload.`
- `service/worker.py`: `TODO: need to design a plan to recover from the annotated video if the service is crashed during the annotate.`
- `service/worker.py`: `TODO: need to find a way to recover from the sns publish if the service is crashed during the sns publish.`
- `service/worker.py`: `TODO: only clean the data but keep the model in GPU/CPU. Let the worker to be`
- `service/worker.py`: `TODO: may need to spread the logic to different steps. No job should be suspended as that wastes the computation resources.`
- `service/worker.py`: `TODO: we need a standard schema for checkpoint data.`
- `service/worker.py`: `TODO: the data shoule be loaded from s3 instead of local file.`

## Rationale
These comments are direct evidence that key lifecycle behaviors (resume semantics, checkpoint schema, crash recovery, and orphan reconciliation) are partially implemented or intentionally deferred. The KB should not present those behaviors as fully complete.

## Impact
- Affects any requirement/decision text that implies production-hardened recovery and resume flow is complete.
- Primary impacted modules: `service/reconciler.py`, `service/routes.py`, `service/worker.py`.

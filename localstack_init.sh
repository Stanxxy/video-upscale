#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# localstack_init.sh — Create LocalStack resources for the video analysis service.
#
# Resources created:
#   SNS  : video_analysis_events               (topic)
#   SQS  : video_analysis_events_dlq           (dead-letter queue)
#   SQS  : video_analysis_events_db_consumer   (main consumer queue, DLQ wired)
#   SNS  : SQS subscription (no filter — receives ALL event types)
#   S3   : bjj-video-analysis                  (output bucket)
#
# Usage:
#   ./localstack_init.sh            — create resources
#   ./localstack_init.sh --status   — check LocalStack health only
#
# Requirements:
#   - LocalStack running on localhost:4566
#   - AWS CLI installed
# ---------------------------------------------------------------------------
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
ENDPOINT="${LOCALSTACK_ENDPOINT:-http://localhost:4566}"
REGION="us-east-1"
ACCOUNT_ID="000000000000"   # LocalStack fixed account ID

TOPIC_NAME="video_analysis_events"
QUEUE_NAME="video_analysis_events_db_consumer"
DLQ_NAME="video_analysis_events_dlq"
S3_BUCKET="bjj-video-analysis"

QUEUE_VISIBILITY_TIMEOUT=60     # seconds — time to process a message before it reappears
QUEUE_RETENTION_SECONDS=86400   # 1 day
QUEUE_WAIT_SECONDS=20           # long-poll receive
DLQ_MAX_RECEIVE_COUNT=3         # move to DLQ after 3 failed deliveries

# Derived ARNs / URLs
TOPIC_ARN="arn:aws:sns:${REGION}:${ACCOUNT_ID}:${TOPIC_NAME}"
QUEUE_ARN="arn:aws:sqs:${REGION}:${ACCOUNT_ID}:${QUEUE_NAME}"
DLQ_ARN="arn:aws:sqs:${REGION}:${ACCOUNT_ID}:${DLQ_NAME}"
QUEUE_URL="${ENDPOINT}/${ACCOUNT_ID}/${QUEUE_NAME}"
DLQ_URL="${ENDPOINT}/${ACCOUNT_ID}/${DLQ_NAME}"

# ── Helpers ─────────────────────────────────────────────────────────────────
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
blue()   { printf '\033[0;34m%s\033[0m\n' "$*"; }
step()   { printf '\n\033[1;34m▶ %s\033[0m\n' "$*"; }
ok()     { printf '  \033[0;32m✓\033[0m %s\n' "$*"; }

# AWS CLI wrapper — always targets LocalStack with dummy credentials
aws_local() {
    AWS_ACCESS_KEY_ID=test \
    AWS_SECRET_ACCESS_KEY=test \
    AWS_DEFAULT_REGION="$REGION" \
    aws --endpoint-url="$ENDPOINT" --region="$REGION" --output json "$@"
}

wait_for_localstack() {
    local max_attempts=15
    local attempt=1
    printf '  Waiting for LocalStack at %s ' "$ENDPOINT"
    while (( attempt <= max_attempts )); do
        if curl -sf "${ENDPOINT}/_localstack/health" > /dev/null 2>&1; then
            printf ' ready\n'
            return 0
        fi
        printf '.'
        sleep 2
        (( attempt++ ))
    done
    printf '\n'
    red "LocalStack did not respond after ${max_attempts} attempts at $ENDPOINT"
    exit 1
}

# ── Status-only mode ─────────────────────────────────────────────────────────
if [[ "${1:-}" == "--status" ]]; then
    echo ""
    blue "LocalStack health ($ENDPOINT):"
    curl -s "${ENDPOINT}/_localstack/health" | python3 -m json.tool 2>/dev/null || \
        red "  LocalStack is not reachable at $ENDPOINT"
    exit 0
fi

# ── Main ─────────────────────────────────────────────────────────────────────
echo ""
blue "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
blue "  LocalStack init — video analysis service"
blue "  Endpoint: $ENDPOINT"
blue "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

step "Checking LocalStack"
wait_for_localstack
ok "LocalStack is healthy at $ENDPOINT"

# ── 1. SNS Topic ─────────────────────────────────────────────────────────────
step "Creating SNS topic: $TOPIC_NAME"
aws_local sns create-topic --name "$TOPIC_NAME" > /dev/null
ok "Topic ARN: $TOPIC_ARN"

# ── 2. Dead-Letter Queue ──────────────────────────────────────────────────────
step "Creating DLQ: $DLQ_NAME"
aws_local sqs create-queue --queue-name "$DLQ_NAME" \
    --attributes MessageRetentionPeriod=604800 > /dev/null
ok "DLQ URL:  $DLQ_URL"
ok "DLQ ARN: $DLQ_ARN"

# ── 3. Main Consumer Queue ────────────────────────────────────────────────────
# AWS CLI v2 --attributes only consumes the first Key=Value as the map; the
# rest become stray positional args.  RedrivePolicy is also JSON-within-JSON.
# Use a temp file (file://) so Python handles all escaping correctly.
step "Creating queue: $QUEUE_NAME"

_QUEUE_ATTRS=$(mktemp)
python3 - <<PYEOF > "$_QUEUE_ATTRS"
import json
policy = {"deadLetterTargetArn": "${DLQ_ARN}", "maxReceiveCount": "${DLQ_MAX_RECEIVE_COUNT}"}
attrs = {
    "VisibilityTimeout":             "${QUEUE_VISIBILITY_TIMEOUT}",
    "MessageRetentionPeriod":        "${QUEUE_RETENTION_SECONDS}",
    "ReceiveMessageWaitTimeSeconds": "${QUEUE_WAIT_SECONDS}",
    "RedrivePolicy":                 json.dumps(policy),
}
print(json.dumps(attrs))
PYEOF

aws_local sqs create-queue --queue-name "$QUEUE_NAME" \
    --attributes "file://${_QUEUE_ATTRS}" > /dev/null
rm -f "$_QUEUE_ATTRS"

ok "Queue URL: $QUEUE_URL"
ok "Queue ARN: $QUEUE_ARN"
ok "DLQ after ${DLQ_MAX_RECEIVE_COUNT} failed receives → $DLQ_NAME"

# ── 4. SQS Resource Policy (allow SNS to publish) ─────────────────────────────
# Policy attribute value must itself be a JSON string (stringified JSON), so we
# need double-encoding — use Python + file:// to avoid shell quoting issues.
step "Setting SQS resource policy"

_SQS_POLICY_ATTRS=$(mktemp)
python3 - <<PYEOF > "$_SQS_POLICY_ATTRS"
import json
policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Sid": "Allow-SNS-SendMessage",
        "Effect": "Allow",
        "Principal": {"Service": "sns.amazonaws.com"},
        "Action": "sqs:SendMessage",
        "Resource": "${QUEUE_ARN}",
        "Condition": {"ArnEquals": {"aws:SourceArn": "${TOPIC_ARN}"}}
    }]
}
# SQS expects Policy attribute value to be a JSON string (double-encoded)
print(json.dumps({"Policy": json.dumps(policy)}))
PYEOF

aws_local sqs set-queue-attributes \
    --queue-url "$QUEUE_URL" \
    --attributes "file://${_SQS_POLICY_ATTRS}" > /dev/null
rm -f "$_SQS_POLICY_ATTRS"
ok "SNS → SQS publish permission set"

# ── 5. Subscribe SQS to SNS ───────────────────────────────────────────────────
# No FilterPolicy = receive ALL event types (bjj_event_detected + analysis_complete)
step "Subscribing $QUEUE_NAME to SNS topic"

SUBSCRIPTION_ARN=$(aws_local sns subscribe \
    --topic-arn "$TOPIC_ARN" \
    --protocol sqs \
    --notification-endpoint "$QUEUE_ARN" \
    --attributes RawMessageDelivery=false \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['SubscriptionArn'])")

ok "Subscription ARN: $SUBSCRIPTION_ARN"
ok "Filter policy: none (receives bjj_event_detected + analysis_complete)"

# ── 6. S3 Output Bucket ───────────────────────────────────────────────────────
step "Creating S3 bucket: $S3_BUCKET"

# us-east-1 does not use LocationConstraint — other regions do
aws_local s3api create-bucket \
    --bucket "$S3_BUCKET" \
    > /dev/null 2>&1 || true   # idempotent — ignore "already exists" on re-run

ok "s3://${S3_BUCKET}"

# ── 7. Verify ─────────────────────────────────────────────────────────────────
step "Verifying resources"

SNS_TOPICS=$(aws_local sns list-topics | python3 -c \
    "import sys,json; ts=json.load(sys.stdin)['Topics']; print(len(ts))")
SQS_QUEUES=$(aws_local sqs list-queues 2>/dev/null | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(len(d.get('QueueUrls',[])))")
S3_BUCKETS=$(aws_local s3api list-buckets | python3 -c \
    "import sys,json; print(len(json.load(sys.stdin)['Buckets']))")

ok "${SNS_TOPICS} SNS topic(s)"
ok "${SQS_QUEUES} SQS queue(s)  (main + DLQ)"
ok "${S3_BUCKETS} S3 bucket(s)"

# ── 8. Print .env snippet ─────────────────────────────────────────────────────
echo ""
blue "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
blue "  Add these to your .env:"
blue "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat <<EOF

BJJ_AWS_REGION=${REGION}
BJJ_AWS_ACCESS_KEY_ID=test
BJJ_AWS_SECRET_ACCESS_KEY=test
BJJ_S3_ENDPOINT_URL=${ENDPOINT}
BJJ_SNS_TOPIC_ARN=${TOPIC_ARN}

EOF
blue "  SQS consumer connection info:"
blue "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat <<EOF

  Queue URL : ${QUEUE_URL}
  Queue ARN : ${QUEUE_ARN}
  DLQ URL   : ${DLQ_URL}
  Region    : ${REGION}
  Endpoint  : ${ENDPOINT}
  Access key: test / test

EOF
green "Done. LocalStack is ready."
echo ""

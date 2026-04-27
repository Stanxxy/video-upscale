---
date: 2026-04-19
category: decision
tags: [infra, aws, production, service, governance]
status: active
---

# AWS Production Infrastructure Baseline (LocalStack Dev-Only)

## Context
Production infrastructure guidance had stale references implying LocalStack and a private endpoint were part of production runtime expectations.

## Content
Production infrastructure for the vision engine is AWS-native. Runtime documentation and operator guidance must treat LocalStack as optional local-development tooling only, never as a production dependency. Production deployment should use AWS credentials/region configuration and pre-provisioned cloud resources.

## Rationale
Mixing local-emulator instructions into production guidance increases onboarding risk and can cause incorrect environment assumptions during incident response or deployment.

## Impact
Updated production-facing docs and operator runbook/skill guidance to remove `http://100.79.167.101:4566` and mark LocalStack as local-dev only. Clarified S3 endpoint behavior in service API/config docs and local setup script defaults.

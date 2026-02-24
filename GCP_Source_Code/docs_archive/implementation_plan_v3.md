# Implementation Plan: Solution Architecture Skill

This plan introduces the **Solution Architect** skill to handle high-level system design, multi-cloud strategies, and integration patterns.

## Proposed Changes

### [NEW] Solution Architect Skill
#### [NEW] [SKILL.md](file:///c:/Users/Ahmed/OneDrive%20-%20Konecta/Documents/mcp/.agent/skills/solution_architect/SKILL.md)
Create a skill focused on "The Big Picture":
- **Cloud Strategy**: AWS vs GCP parity (Lambda vs Cloud Run, S3 vs GCS).
- **Architecture Patterns**: Monolith to Microservices, Event-Driven Design, Serverless.
- **Integration**: Pub/Sub, API Gateway, Message Queues (Kafka/RabbitMQ).
- **Data Architecture**: Consistency vs Availability (CAP Theorem), Database selection.
- **Security & Governance**: Secret management, IAM least privilege, Cloud Compliance.

### [MODIFY] Workspace Standards
#### [MODIFY] [antigravity-specs.md](file:///c:/Users/Ahmed/OneDrive%20-%20Konecta/Documents/mcp/.amazonq/rules/antigravity-specs.md)
- Ensure all specifications include a "Cloud Parity" section for multi-cloud deployments.

---

## Verification Plan

### Automated Verification
- Verify the skill file exists and correctly triggers for architectural prompts.

### Manual Verification
- Review the skill content for bilingual (English/Arabic) consistency and alignment with Antigravity's Lead Developer philosophy.

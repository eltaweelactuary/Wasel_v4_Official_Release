# Documentation: Estimated Project Costs (Phase 1)

**Project:** Wasel v4 Pro  
**Stage:** Phase 1 (Financial Overview)

---

## ðŸ’° GCP Infrastructure Costs (Estimated)

Because Wasel v4 Pro uses a **Serverless Architecture** (Cloud Run), costs are primarily based on usage rather than fixed monthly fees.

| Service | Estimated Usage (MVP) | Monthly Cost (Est.) |
|---|---|---|
| **Google Cloud Run** | 180k vCPU-seconds / month | **$0.00** (Free Tier) |
| **Cloud Build** | 2,500 build-minutes / month | **$0.00** (First 120min/day Free) |
| **Cloud Storage (GCS)** | 5 GB storage + 5,000 Class A ops | **<$0.50** |
| **Networking** | 10 GB Egress | **<$1.20** |
| **TOTAL (Phase 1)** | | **~$2 - $5 / month** |

### Cost Optimization Features
*   **Scale-to-Zero:** When no one is using the app, Cloud Run instances are shut down, resulting in $0 cost.
*   **Static Assets:** Landmarks and DNA are small (<50KB per word), keeping storage and transfer costs minimal.

---

## ðŸ“ˆ Long-term Scalability (Production)
For a larger user base (e.g., 1,000 active daily users), the estimated cost would scale linearly but remain highly efficient:
*   **Estimated Production Cost:** ~$25 - $40 / month.

---
> [!NOTE]
> These are estimates based on standard Google Cloud pricing. Actual costs may vary depending on regions and specific usage patterns.


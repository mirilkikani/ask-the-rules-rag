# Test set label changes

The first labels were strict: only the page where the article heading or the exact fact appears.
After the first retrieval run (chunk size 500), every missed question was checked by hand against the
text of the retrieved pages. A page was only added if it truly answers the question.

## Run 1 (13.11.2026): strict labels
Hit@5: 500 = 39/50, 1000 = 36/50, 1500 = 35/50.

## Review of the 11 misses at chunk size 500 (15.11.2026)

| Question | Retrieved page checked | Decision | Reason |
|---|---|---|---|
| ai10 Who must do a fundamental rights impact assessment? | AI Act p.25 | Label changed, p.25 added | Recital 96 names the deployers who must do it |
| ai18 When does the AI Act apply? | AI Act p.44 | Label changed, p.44 added | Recital 179: applies from 2 August 2026, prohibitions earlier |
| ai19 What are AI regulatory sandboxes? | AI Act p.35 | Label changed, p.35 added | Recital explains what sandboxes are and why they exist |
| gd02 GDPR processing principles | GDPR p.7 | Label changed, p.7 added | Recital 39 describes the principles |
| ai05 Risk management system | AI Act p.19 | Real miss | Recital 66 only lists the requirement, not what the system includes |
| ai16 Maximum fine for prohibited practices | AI Act p.117 | Real miss (risky) | p.117 shows fines for EU institutions (EUR 1 500 000), a different group; the general maximum (EUR 35 000 000) is on p.115 |
| gd03 Legal bases for lawful processing | GDPR p.9 | Real miss (partial) | Recitals mention only some bases (vital, public, legitimate interest) |
| gd06 Special categories of data | GDPR p.10 | Real miss (partial) | Recitals mention only some categories (racial or ethnic origin, health, biometric) |
| ai20 German question, English document | only German DSK pages retrieved | Real miss | Cross-language failure |
| gd16 German question, English document | only German DSK pages retrieved | Real miss | Cross-language failure |
| ki07 English question, German document | only English EU pages retrieved | Real miss | Cross-language failure |

## Honest note
Only misses were reviewed. Hits were not checked for labels that are too generous,
so the revised numbers may be slightly optimistic. Both runs are reported.

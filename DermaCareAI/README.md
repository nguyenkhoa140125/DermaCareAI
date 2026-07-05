# DermaCare AI v1.0
## Dermatology Health Intelligence Platform

## CRITICAL SKIN EMERGENCY DISCLAIMER
- **Anaphylaxis** (rash + throat swelling/breathing difficulty): IM adrenaline + **112/999/911 IMMEDIATELY**
- **SJS/TEN** (skin peeling over large areas + fever): Hospital/burns unit IMMEDIATELY, stop causative drug
- **Necrotising fasciitis** (spreading dusky skin + pain + fever): SURGICAL EMERGENCY, 112/999/911
- **Erythroderma** (90%+ BSA erythema + systemic symptoms): Hospital admission IMMEDIATELY
- All content is AI-generated educational research only — NOT medical advice

## Quick Start (Windows)
1. Extract ZIP to any folder
2. Double-click **START_DermaCare_AI.bat**
3. Auto-installs everything (2-5 min first time)
4. Browser opens at http://localhost:5090
5. Accept disclaimer and begin

## Security — AES-256-GCM
- API keys AES-256-GCM encrypted client-side before localStorage
- PBKDF2 key derivation (100,000 iterations) from device fingerprint
- Keys never logged by backend
- Backend rate limiting (30 req/60s), input sanitisation, provider whitelist

## AI Providers (5 Options)
| Provider | Model | Get Key |
|---|---|---|
| Claude (Anthropic) | claude-sonnet-4 | console.anthropic.com |
| ChatGPT (OpenAI) | gpt-4o | platform.openai.com/api-keys |
| Gemini (Google) | gemini-2.0-flash | aistudio.google.com/apikey |
| Grok (xAI) | grok-2-latest | console.x.ai |
| DeepSeek | deepseek-chat | platform.deepseek.com/api_keys |

## Sections Covered (13 Panels)
- **Conditions** — 35+ via dropdown (Inflammatory, Acne, Infections, Urticaria, Pigmentation, Hair, Cancer)
- **Acne** — 4 tabs: Overview/Grading, Treatment Ladder, Isotretinoin (IPPA/teratogenicity), Scarring/PIH
- **Psoriasis** — 5 tabs: Types (plaque/guttate/inverse/pustular/erythrodermic), PASI/DLQI scoring, Topical/Phototherapy, Systemic/Biologics (IL-17/IL-23/TYK2), Psoriatic Arthritis
- **Skin Cancer** — 5 tabs: Melanoma (ABCDE, staging, immunotherapy), BCC (Mohs, hedgehog inhibitors), SCC+AK (field treatment), Dermoscopy (two-step algorithm), Skin cancer in dark skin
- **Treatments** — 5 tabs: Topical agents table, Biologics table (dupilumab/secukinumab/risankizumab), JAK inhibitors (baricitinib/upadacitinib/ruxolitinib/deucravacitinib), Phototherapy (NB-UVB/PUVA/PDT), Procedures
- **Skincare** — Personalised assessment form + SPF science + active ingredients evidence table
- **Hair & Nails** — 4 tabs: Alopecia Areata (JAK inhibitors), Androgenetic Alopecia, Telogen Effluvium, Nail Disorders
- **Emergency** — Anaphylaxis, SJS/TEN, Necrotising Fasciitis, DRESS, Erythroderma
- **Skin of Colour** — Fitzpatrick scale table, PIH, Keloids, Eczema in dark skin, Psoriasis in dark skin, Melasma, AKN, Afro-Caribbean hair disorders

## India Dermatology Resources
- IADVL: iadvl.org | AIIMS Dermatology: aiims.edu
- Apollo/Fortis Dermatology | Emergency: 112

## Clinical Sources
AAD | BAD | EADV | WHO | NICE | IADVL | PubMed

*DermaCare AI v1.0 — For research and educational purposes only. Not medical advice.*
*SKIN EMERGENCY: 112 (India) / 999 (UK) / 911 (US)*

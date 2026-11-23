"""
DermaCare AI - Production Backend Server v1.0
Dermatology Health Intelligence Platform
=========================================
DISCLAIMER: All AI output is for research/education only.
Not medical advice. Always consult a qualified dermatologist.
SKIN EMERGENCY: Sudden widespread rash with fever/difficulty breathing
(anaphylaxis/SJS/TEN), rapidly spreading skin infection with fever,
infected wound with red streaks - Call 112 (India) / 999 (UK) / 911 (US).
"""

import os, sys, json, uuid, time, hashlib, logging, datetime, argparse
from pathlib import Path

try:
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
except ImportError:
    print("[FATAL] Flask not installed. Run REPAIR_AND_RECOVER.bat"); sys.exit(1)

try:
    import requests as req_lib; REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    import fitz; FITZ_OK = True
except ImportError:
    FITZ_OK = False

try:
    from PIL import Image; PIL_OK = True
except ImportError:
    PIL_OK = False

sys.path.insert(0, str(Path(__file__).parent / "modules"))
try:
    import ai_providers; AI_PROVIDERS_OK = True
except ImportError:
    AI_PROVIDERS_OK = False

BASE_DIR    = Path(__file__).parent.resolve()
UPLOAD_DIR  = BASE_DIR / "uploads"
LOGS_DIR    = BASE_DIR / "logs"
DATA_DIR    = BASE_DIR / "data"
STATIC_DIR  = BASE_DIR / "static"
REPORTS_DIR = BASE_DIR / "reports_db"

for d in [UPLOAD_DIR, LOGS_DIR, DATA_DIR, STATIC_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PORT    = int(os.environ.get("DERMACARE_PORT", 5090))
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DEFAULT_PROVIDER_KEYS = ai_providers.get_env_keys() if AI_PROVIDERS_OK else {}
VERSION = "1.0.0"

DISCLAIMER = (
    "WARNING - AI RESEARCH DISCLAIMER: All output is AI-generated from published "
    "dermatology literature (AAD, BAD, EADV, WHO, NICE, IADVL, PubMed). This is for "
    "educational research only. NOT a medical diagnosis or prescription. ALWAYS consult "
    "a qualified dermatologist before any skin health decision. DERMATOLOGY EMERGENCY "
    "(sudden widespread rash with fever/throat swelling, skin peeling over large areas, "
    "rapidly spreading red infection with fever): "
    "Call 112 (India) / 999 (UK) / 911 (US) immediately."
)

log_file = LOGS_DIR / f"server_{datetime.date.today()}.log"
handlers = [logging.StreamHandler(sys.stdout)]
try:
    handlers.insert(0, logging.FileHandler(log_file, encoding="utf-8"))
except Exception:
    pass
logging.basicConfig(level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    handlers=handlers)
log = logging.getLogger("DermaCareAI")

app = Flask(__name__, static_folder=str(STATIC_DIR))
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
CORS(app, origins="*")

_RATE_STORE = {}

def _get_client_id():
    return hashlib.sha256((request.remote_addr or '127.0.0.1').encode()).hexdigest()[:16]

def rate_limit_check():
    cid = _get_client_id(); now = time.time()
    _RATE_STORE.setdefault(cid, [])
    _RATE_STORE[cid] = [t for t in _RATE_STORE[cid] if now - t < 60]
    if len(_RATE_STORE[cid]) >= 30: return False
    _RATE_STORE[cid].append(now); return True

# --- Khoanh vung nam (melasma) ---
try:
    from melasma_seg import register as register_melasma
    register_melasma(app, rate_limit_check)
    log.info("Da gan /api/melasma/segment")
except Exception as e:
    log.warning("Khong gan duoc module melasma: %s", e)

def sanitise_api_key(key):
    if not key or not isinstance(key, str): return ''
    key = key.strip()
    if len(key) > 512: return ''
    s = ''.join(c for c in key if 0x21 <= ord(c) <= 0x7E)
    return s if len(s) >= 10 else ''

def validate_provider(p):
    return p.lower() if p and p.lower() in {'anthropic','openai','gemini','grok','deepseek'} else 'anthropic'

# ===== DERMATOLOGY KNOWLEDGE BASE =====
KNOWLEDGE = {
    "eczema_dermatitis": {
        "name": "Eczema and Dermatitis",
        "atopic_dermatitis": "Atopic dermatitis (atopic eczema) is the most common chronic inflammatory skin condition, affecting up to 20% of children and 5-10% of adults globally. It is part of the atopic triad (eczema, asthma, allergic rhinitis). Characterised by intensely itchy, dry, inflamed skin with flexural distribution (elbow and knee creases, neck, wrists, ankles) in older children and adults, and face/scalp/extensor surfaces in infants. Caused by defective skin barrier (filaggrin gene mutations in many cases) allowing allergen sensitisation, immune dysregulation (Th2-skewed in acute phase), and secondary infections (Staphylococcus aureus colonises nearly all moderate-severe AD).",
        "management_steps": "Step 1 (Mild): Emollients liberally and frequently (cornerstone of all AD management - at least 250g/week for an adult), gentle cleansers, trigger avoidance. Step 2 (Mild-Moderate): Topical corticosteroids (TCS) for flares - mild potency (hydrocortisone 1%) for face/groin/axillae, moderate potency (betamethasone valerate 0.025%, clobetasone 0.05%) for body, potent (betamethasone 0.1%, mometasone) for thickened/palmar/plantar. Topical calcineurin inhibitors (tacrolimus 0.03/0.1%, pimecrolimus 1%) for face/skin folds as steroid-sparing. Step 3 (Moderate-Severe): Dupilumab (IL-4/IL-13 inhibitor) - biologic, SC injection every 2 weeks, highly effective, no systemic immunosuppression, approved from age 6 months. Tralokinumab (anti-IL-13). Step 4 (Severe refractory): JAK inhibitors (upadacitinib, abrocitinib, baricitinib - oral, rapid onset). Ciclosporin (short-term rescue), methotrexate, azathioprine (conventional immunosuppressants with monitoring requirements).",
        "contact_dermatitis": "Allergic contact dermatitis (ACD): delayed (Type IV) hypersensitivity to hapten. Common allergens: nickel (jewellery, belt buckles), rubber chemicals/latex, fragrances (personal care products), preservatives (methylisothiazolinone/methylchloroisothiazolinone - MI/MCI in wet wipes), chromate (cement, leather), paraphenylenediamine (hair dye - PPD). Diagnosis: patch testing (gold-standard) - allergen panels applied to back for 48 hours, readings at 48h and 96h. Management: identify and strictly avoid causative allergen, topical corticosteroids for acute phase, barrier protection. Irritant contact dermatitis (ICD): non-immunological damage from repeated exposure to irritants (detergents, water, chemicals) - most common occupational skin disease (healthcare workers, hairdressers, cleaners).",
        "seborrhoeic_dermatitis": "Chronic relapsing condition affecting sebaceous-gland-rich areas (scalp, face - nasolabial folds, eyebrows, glabella, ears, central chest). Associated with Malassezia yeast overgrowth and sebaceous activity. Erythematous patches with greasy yellow scales. Dandruff is the scalp-only presentation. Management: antifungal shampoos (ketoconazole 2%, zinc pyrithione, selenium sulphide, ciclopirox) - apply and leave 5 minutes before rinsing. Facial seborrhoeic dermatitis: ketoconazole 2% cream, low-potency topical steroids (short-term), topical calcineurin inhibitors for steroid-sparing maintenance.",
        "emollients": "Emollients are the cornerstone of all dry skin and eczema management. Types: lotions (lightest, mainly water - less effective for severe dry skin), creams (oil-water emulsion - good general use), ointments (greasy, most occlusive - best for very dry/lichenified skin, but poorly tolerated in hot weather). Apply immediately after bathing (within 3 minutes - 'soak and seal'). Apply in downward strokes in direction of hair growth (reduces folliculitis risk). Prescribe generously and repeat throughout the day. Avoid aqueous cream as a leave-on emollient (contains sodium lauryl sulphate which damages skin barrier).",
    },
    "psoriasis": {
        "name": "Psoriasis",
        "definition": "Chronic, immune-mediated inflammatory skin condition affecting approximately 2-3% of the global population. Characterised by accelerated keratinocyte proliferation (turnover reduced from 28 days to 3-4 days) producing well-demarcated, erythematous plaques with silvery-white scale. Strong genetic component (multiple susceptibility loci including HLA-Cw6). Systemic condition - associated with psoriatic arthritis (25-30%), cardiovascular disease, metabolic syndrome, inflammatory bowel disease, and depression.",
        "clinical_types": "Plaque psoriasis (psoriasis vulgaris, approximately 85%): well-demarcated erythematous plaques with thick silvery scale, typically on elbows, knees, scalp, lower back. Guttate psoriasis: small teardrop-shaped lesions on trunk/limbs, classically triggered by streptococcal throat infection, common in young adults/children. Inverse/flexural psoriasis: affects skin folds (axillae, groin, submammary, natal cleft) - minimal scale due to moisture, risk of misdiagnosis as fungal infection or intertrigo. Pustular psoriasis: sterile pustules - localised (palmoplantar pustulosis, PPP) or generalised (von Zumbusch - EMERGENCY with fever, systemic toxicity). Erythrodermic psoriasis: affects 90%+ body surface area - EMERGENCY with temperature dysregulation and systemic effects.",
        "nail_psoriasis": "Affects up to 50% of psoriasis patients, 80% of those with psoriatic arthritis. Features: pitting (multiple punctate depressions - most specific), onycholysis (nail plate lifting from nail bed), subungual hyperkeratosis, oil drop/salmon patch sign (yellow-brown discoloration), leukonychia, nail dystrophy. Nail psoriasis predicts psoriatic arthritis and is important to assess. Difficult to treat - topical steroids/calcipotriol under occlusion, intralesional steroid injections, biologics for severe cases.",
        "psoriatic_arthritis": "Inflammatory arthropathy in 25-30% of psoriasis patients. May precede skin disease in some. Patterns: distal interphalangeal (DIP) joint predominant, asymmetric oligoarthritis, symmetric polyarthritis (RA-like), spondyloarthritis (axial involvement), arthritis mutilans (destructive, rare). Dactylitis (sausage digit) and enthesitis (inflammation at tendon/ligament insertions, especially Achilles/plantar fascia) are hallmarks. Manage jointly with rheumatology. Biologics (anti-TNF, IL-17 inhibitors, IL-23 inhibitors, JAK inhibitors) treat both skin and joint disease.",
        "treatments": "Topical (mild-moderate): vitamin D analogues (calcipotriol/calcitriol - reduce keratinocyte proliferation, no skin atrophy), topical corticosteroids, combination calcipotriol+betamethasone (Enstilar foam/Dovobet gel - convenient, effective), coal tar (anti-proliferative, anti-inflammatory, particularly for scalp/body), dithranol (short-contact therapy, stains). Phototherapy (moderate-severe): narrowband UVB (NB-UVB) - most used phototherapy, 3x/week clinic visits; PUVA (psoralen + UVA, more effective but higher carcinogenicity risk). Conventional systemic (moderate-severe): methotrexate (folic acid supplementation mandatory), ciclosporin (short-term rescue), acitretin (for pustular/erythrodermic, teratogenic - very strict pregnancy precautions). Biologics (moderate-severe, failed conventional): TNF inhibitors (adalimumab, etanercept, infliximab), IL-12/23 inhibitor (ustekinumab), IL-17 inhibitors (secukinumab, ixekizumab, bimekizumab - fastest skin clearance), IL-23 inhibitors (guselkumab, risankizumab, tildrakizumab - convenient dosing, durable remission), JAK inhibitors (deucravacitinib - TYK2 inhibitor, oral, favorable safety profile vs pan-JAK inhibitors).",
        "pasi_score": "Psoriasis Area and Severity Index (PASI) is the standard severity measure used clinically and in research. PASI 75 (75% improvement) is the traditional benchmark for treatment efficacy. PASI 90 (90% improvement) is the modern standard with biologics. PASI 100 (complete clearance) is achievable with IL-17 and IL-23 inhibitors. DLQI (Dermatology Life Quality Index) assesses quality of life impact. Moderate-severe psoriasis defined as BSA over 10%, PASI over 10, or DLQI over 10.",
    },
    "acne": {
        "name": "Acne Vulgaris",
        "pathophysiology": "Multifactorial condition involving four key pathogenic factors: (1) excess sebum production (androgen-driven, DHT via 5-alpha reductase), (2) follicular hyperkeratinisation (abnormal desquamation of follicular epithelium causing comedone formation), (3) Cutibacterium acnes (C. acnes, formerly P. acnes) colonisation and biofilm formation, (4) inflammation (both innate and adaptive immune responses). Results in comedones (open blackheads, closed whiteheads), papules, pustules, nodules, and cysts.",
        "grading": "Mild: mainly comedones, few papules/pustules, no nodules. Moderate: multiple papules and pustules, few nodules, involving face +/- trunk. Severe: extensive papules/pustules, multiple nodules/cysts, significant inflammation - risk of permanent scarring. NODULOCYSTIC acne: large painful cysts and nodules, high scarring risk - requires systemic treatment.",
        "treatments": "Topical first-line: benzoyl peroxide (BPO) 2.5-10% - bactericidal, anti-inflammatory, reduces resistance risk; topical retinoids (tretinoin 0.025-0.1%, adapalene 0.1-0.3%, tazarotene) - normalise follicular desquamation, anti-comedogenic, anti-inflammatory, first-line for comedonal acne; topical antibiotics (clindamycin 1%, erythromycin) - ALWAYS combine with BPO to prevent resistance; azelaic acid (15-20%) - anti-inflammatory, anti-comedogenic, good for PIH/rosacea, safe in pregnancy. Fixed-dose combinations (adapalene+BPO, clindamycin+BPO) improve compliance. Systemic: oral antibiotics (doxycycline 100mg OD/BD, lymecycline 408mg OD - limit to 3 months to reduce resistance, always add BPO); combined oral contraceptive pill (COCP - co-cyprindiol/Dianette, ethinylestradiol+norgestimate) for females with hormonal acne; spironolactone 25-200mg for adult female hormonal acne (off-label UK, licensed in some countries); oral isotretinoin (Roaccutane) for severe/nodulocystic acne or moderate acne failing other treatments - very effective but teratogenic (pregnancy prevention programme mandatory), side effects: dry lips/skin, elevated triglycerides/LFTs, mood monitoring, IPledge/IPPA programme compliance required.",
        "scarring": "Acne scarring is permanent and psychologically devastating. Prevention by early adequate treatment is key. Types: atrophic (most common) - icepick (narrow deep), boxcar (broad flat base), rolling (sloping edges); hypertrophic/keloidal (raised). Treatments for established scarring: chemical peels (glycolic/salicylic/TCA - superficial), microneedling/radiofrequency microneedling, fractional laser (ablative CO2 or non-ablative), subcision (for rolling scars), dermal fillers (temporary improvement), punch excision/elevation (icepick scars). Combination approaches give best results. Realistic expectation-setting important.",
        "post_inflammatory": "Post-inflammatory hyperpigmentation (PIH) is extremely common, especially in Fitzpatrick skin types IV-VI (darker skin tones). Not true scarring but can persist months-years. Management: sun protection (SPF 30+ daily - essential), topical retinoids, azelaic acid, vitamin C serum, niacinamide, hydroquinone 2-4% (short courses), kojic acid. Lasers/chemical peels must be used very cautiously in darker skin tones (risk of PIH paradoxically worsening). Treat the underlying acne first.",
    },
    "skin_cancer": {
        "name": "Skin Cancer and Pigmented Lesions",
        "melanoma": "Most dangerous skin cancer. Arises from melanocytes. Risk factors: fair skin (Fitzpatrick types I-II), UV exposure (sunburn history especially blistering burns in childhood), family history, multiple/atypical naevi, giant congenital melanocytic naevus, CDKN2A/CDK4 mutations, immunosuppression. ABCDE dermoscopy/clinical criteria: Asymmetry, Border irregularity, Colour variation (multiple colours), Diameter over 6mm, Evolution (changing). Subtypes: superficial spreading (most common, 70%), nodular (most aggressive, rapidly growing), lentigo maligna (elderly, sun-damaged skin, slow growing), acral lentiginous (palms/soles/subungual, common in darker skin). Staging: Breslow thickness (depth in mm) most important prognostic factor; Clark level (anatomical depth); sentinel lymph node biopsy for lesions over 0.8mm. Treatment: wide local excision (margins 1-2cm depending on Breslow), SLNB for staging, adjuvant immunotherapy (nivolumab, pembrolizumab) or targeted therapy (BRAF+MEK inhibitors for BRAF V600E mutant - vemurafenib+cobimetinib, dabrafenib+trametinib) for high-risk resected/advanced disease.",
        "bcc": "Basal Cell Carcinoma (BCC): most common skin cancer globally (approximately 80% of non-melanoma skin cancers). Slow growing, locally invasive, rarely metastasises. Risk factors: UV exposure (main cause), fair skin, ionising radiation, arsenic, immunosuppression, Gorlin syndrome (PTCH1 mutation). Clinical types: nodular (pearly papule/nodule with telangiectasiae, rolled border - classic), superficial (erythematous patch, may resemble eczema), morphoeic/sclerosing (scar-like, ill-defined - hardest to treat, aggressive), pigmented (in darker skin types), basosquamous (higher metastatic potential). Treatment: surgical excision (gold-standard, clear histological margins), Mohs micrographic surgery (for high-risk/facial/recurrent BCC - highest cure rate with maximum tissue conservation), topical treatments for superficial BCC (imiquimod 5% cream, photodynamic therapy/PDT), radiotherapy (elderly/inoperable). Advanced/metastatic: hedgehog pathway inhibitors vismodegib or sonidegib. Prognosis excellent if treated early.",
        "scc": "Squamous Cell Carcinoma (SCC): second most common skin cancer, approximately 20% of NMSC. Arises from keratinocytes. Risk factors: cumulative UV exposure (main factor, unlike BCC which is peak UV exposure), fair skin, immunosuppression (transplant recipients - dramatically elevated risk, more aggressive disease), HPV (especially anogenital SCC), actinic keratoses (precursor lesions), chronic wounds/scars (Marjolin's ulcer). Clinical features: indurated erythematous nodule or plaque that may ulcerate, keratinous scale/crust, often on sun-exposed sites (scalp, ears, lips, dorsal hands). High-risk features: size over 2cm, depth over 2mm, poor differentiation, perineural invasion, immunosuppression, lip/ear/genital site. Treatment: surgical excision with minimum 4mm margins (wider for high-risk), Mohs for high-risk sites, radiotherapy, sentinel lymph node biopsy for high-risk, cetuximab or PD-1 inhibitors (cemiplimab, pembrolizumab) for advanced/metastatic disease.",
        "actinic_keratosis": "Actinic keratoses (AK, solar keratoses) are pre-malignant lesions - dysplastic keratinocytes confined to epidermis. Risk of progression to invasive SCC is individually low (approximately 0.025-0.16% per lesion per year) but cumulative risk with multiple lesions is significant. Clinical features: rough, scaly, erythematous papules or patches on sun-damaged skin. Feldisation (field cancerisation): widespread subclinical sun damage in surrounding skin. Field treatment preferred over single-lesion treatment: 5-fluorouracil cream (Efudix 5% - intense local inflammatory reaction, effective, short course), imiquimod 5% cream (immune-stimulating, 3x/week for 4-12 weeks), ingenol mebutate (withdrawn in Europe due to cancer safety concerns), diclofenac 3% gel (mild, suitable for sensitive patients), photodynamic therapy (PDT - most effective for thick/hyperkeratotic lesions, cosmetically excellent result).",
        "dermoscopy": "Dermoscopy (dermatoscopy) uses a handheld dermoscope with polarised light to visualise subsurface skin structures not visible to the naked eye, dramatically improving diagnostic accuracy for pigmented lesions. Key structures assessed: pigment network, atypical vessels, regression structures, blue-white veil, globules, streaks, pseudopods. The two-step algorithm: (1) differentiate melanocytic from non-melanocytic; (2) if melanocytic, use pattern analysis/ABCD rule/7-point checklist to determine benign vs malignant. ABCD dermoscopic rule (different from clinical ABCDE): Asymmetry, Border, Colour, Dermoscopic structures.",
    },
    "infections": {
        "name": "Skin Infections",
        "bacterial": "Impetigo: superficial bacterial skin infection, most commonly Staphylococcus aureus (+/- Streptococcus pyogenes). Highly contagious. Non-bullous impetigo: honey-coloured/golden crusted lesions, typically on face around nose/mouth. Bullous impetigo: S. aureus toxin-producing strains causing blistered lesions. Management: localised - topical fusidic acid or mupirocin 2%; widespread - oral flucloxacillin (first-line UK) or cefalexin. MRSA: consult local guidelines. Cellulitis/Erysipelas: acute bacterial skin and soft tissue infection. Erysipelas is more superficial (dermis/upper subcutaneous) with sharply demarcated raised border. Cellulitis: deeper dermis/subcutaneous fat, ill-defined border. Streptococcus pyogenes most common cause; S. aureus in cases with wound/trauma. Management: elevation, marking border to monitor spread, oral flucloxacillin/cefalexin (most cases), IV benzylpenicillin or IV flucloxacillin if systemically unwell (Eron Class III/IV). EMERGENCY: necrotising fasciitis - rapidly spreading infection with severe pain, skin necrosis, systemic toxicity, subcutaneous gas on imaging - surgical emergency requiring immediate debridement.",
        "fungal": "Dermatophyte infections (tinea/ringworm): caused by Trichophyton, Microsporum, Epidermophyton species. Named by body site: tinea capitis (scalp), tinea corporis (body), tinea pedis (athlete's foot - most common fungal infection), tinea cruris (groin - jock itch), tinea unguium/onychomycosis (nails), tinea manuum (hands). Diagnosis: skin scrapings for microscopy and culture (KOH preparation shows fungal hyphae); Wood's lamp (Microsporum canis fluoresces blue-green). Management: topical antifungals for limited disease (clotrimazole, terbinafine, miconazole, bifonazole); oral terbinafine 250mg OD (most effective for dermatophytes, including nails - 3 months toenails, 6 weeks fingernails) or itraconazole for nail disease; scalp disease (tinea capitis) requires oral antifungals (griseofulvin/itraconazole) as topical agents alone do not penetrate. Pityriasis versicolor: Malassezia furfur - hypo/hyperpigmented macules, trunk/shoulders, fine scale. Ketoconazole/selenium sulphide shampoo left on, or single-dose fluconazole/itraconazole.",
        "viral": "Herpes simplex: HSV-1 (mainly oral/labialis) and HSV-2 (mainly genital, though HSV-1 now responsible for 50%+ of primary genital herpes). Primary infection often subclinical or severe with widespread vesicles. Recurrent episodes triggered by stress, sunlight, illness, immunosuppression. Management: aciclovir 200mg 5x/day for 5 days or valaciclovir 500mg BD. Herpes zoster (shingles): VZV reactivation from dorsal root ganglion. Dermatomal unilateral vesicular rash with pain (often preceding rash by several days). Postherpetic neuralgia (PHN) is the most common complication (risk increases with age). Treatment: antiviral within 72 hours of rash onset reduces severity/PHN risk (aciclovir 800mg 5x/day, valaciclovir 1g TDS, famciclovir 250mg TDS for 7 days). PHN: pregabalin, gabapentin, amitriptyline, topical lidocaine/capsaicin patches. Vaccination: live attenuated Zostavax (60+) or recombinant Shingrix (50+ - more effective, preferred). Warts: HPV infection. Verruca vulgaris (common wart), verruca plana (flat warts), plantar wart (verruca). Most resolve spontaneously in healthy individuals. Treatments: cryotherapy (liquid nitrogen), salicylic acid preparations, silver nitrate, immunotherapy (DNCB, squaric acid dibutylester), bleomycin injection for resistant warts. Molluscum contagiosum: poxvirus, small pearlescent umbilicated papules, self-limiting in immunocompetent individuals.",
        "necrotising_fasciitis": "DERMATOLOGICAL/SURGICAL EMERGENCY. Rapidly progressive infection of the deep fascia with secondary necrosis of subcutaneous tissue. Mortality 20-30%. Type I: polymicrobial (diabetics, immunosuppressed, abdominal wounds). Type II: group A Streptococcus (pyogenes). Type III: Clostridial (gas gangrene). Early signs: severe pain disproportionate to appearance, rapid spread, oedema, skin colour changes (dusky/grey), blistering. Late signs: skin necrosis, crepitus (subcutaneous gas), systemic sepsis. Management: IMMEDIATE surgical debridement (do not delay for imaging), broad-spectrum IV antibiotics, HDU/ICU, IV immunoglobulin for GAS type II.",
    },
    "rosacea": {
        "name": "Rosacea",
        "definition": "Chronic inflammatory facial condition predominantly affecting fair-skinned adults, typically presenting in the 30s-50s, more common in women but often more severe in men. Four subtypes (Wilkin classification): ETR (erythematotelangiectatic - flushing, persistent central facial erythema, telangiectasia), PPR (papulopustular - central facial inflammatory papules and pustules, resembles acne), phymatous (skin thickening/hyperplasia, predominantly nose - rhinophyma, mainly men), ocular (eyelid inflammation, conjunctivitis, dry eye - affects up to 50% of rosacea patients, may be the presenting feature).",
        "triggers": "UV light, heat, hot/spicy foods and drinks, alcohol (especially red wine), exercise, cold/wind, stress, certain skincare products (fragrances, astringents, alcohol-based), Demodex folliculorum mite overgrowth (implicated in pathogenesis), certain medications (vasodilators). Avoid triggers as primary management strategy.",
        "management": "Skincare: gentle non-irritating cleanser, moisturiser to repair barrier function, broad-spectrum mineral sunscreen (SPF 30+, zinc oxide/titanium dioxide preferred as chemical sunscreens can irritate). Topical treatments: metronidazole 0.75-1% cream/gel (anti-inflammatory, first-line for PPR), azelaic acid 15-20% (anti-inflammatory, anti-Demodex, also reduces PIH - good tolerated choice), ivermectin 1% cream (anti-Demodex, superior to metronidazole in trials), brimonidine 0.33% gel (alpha-agonist, reduces erythema within 30 minutes - 'on-demand' use), oxymetazoline 1% cream (alpha-agonist for persistent erythema, once daily). Oral treatments: doxycycline 40mg MR (Efracea - sub-antimicrobial dose, anti-inflammatory only, UK licensed) or doxycycline 100mg for moderate-severe PPR; oral ivermectin (Demodex-driven disease). Laser/IPL: excellent for telangiectasia and erythema (pulsed dye laser, Nd:YAG, intense pulsed light). Rhinophyma: surgical debulking, CO2 laser ablation.",
        "ocular_rosacea": "Often undiagnosed. Symptoms: burning, stinging, gritty sensation, photophobia, blurred vision. Signs: lid margin telangiectasia, meibomian gland dysfunction (MGD), chalazia, conjunctival injection, corneal involvement (keratitis) in severe cases. Management: warm compresses and lid hygiene (twice daily), artificial tears, topical metronidazole/azithromycin to eyelid margins, oral doxycycline for moderate-severe cases. Referral to ophthalmology if corneal involvement suspected.",
    },
    "urticaria_angioedema": {
        "name": "Urticaria and Angioedema",
        "urticaria": "Urticaria (hives): transient, migratory, intensely pruritic wheals (raised, oedematous plaques surrounded by erythema) that individually last under 24 hours, blanch on pressure, and leave no residual mark. Acute urticaria (under 6 weeks): commonly allergic (IgE-mediated - foods, drugs, stings), viral, or idiopathic. Chronic urticaria (over 6 weeks): most commonly chronic spontaneous urticaria (CSU, previously called chronic idiopathic urticaria) - autoimmune mechanism in many (IgG autoantibodies against IgE receptor or IgE). Investigation of acute: identify trigger (IgE testing if suspected). Investigation of chronic: CBC, ESR/CRP, thyroid function/autoantibodies, autologous serum skin test/basophil activation test. Management: second-generation non-sedating H1 antihistamines (cetirizine, loratadine, fexofenadine, bilastine) - first-line, may require updosing up to 4x standard dose; omalizumab (anti-IgE, SC injection monthly) for antihistamine-refractory CSU - highly effective; ciclosporin for omalizumab-refractory.",
        "angioedema": "Deep dermal and subcutaneous/submucosal oedema affecting face (periorbital, lips), tongue, throat, hands, feet, genitalia. Can be life-threatening if laryngeal/pharyngeal involvement causes airway obstruction. Classified: allergic/histaminergic (rapid onset, urticaria usually present, responds to antihistamines/adrenaline), C1-inhibitor deficiency hereditary angioedema (HAE - autosomal dominant, C1-inhibitor deficient/dysfunctional, bradykinin-mediated, NOT histamine - does NOT respond to antihistamines/steroids/adrenaline - different mechanism and treatment), ACE inhibitor-induced (bradykinin accumulation, onset can be years after starting ACE-I, switch to ARB).",
        "anaphylaxis": "MEDICAL EMERGENCY. Rapid-onset life-threatening systemic allergic reaction. Triggers: foods (peanut, tree nuts, shellfish, fish, milk, egg), drugs (penicillin, NSAIDs, contrast media), stings, latex. Clinical features: urticaria/angioedema + respiratory symptoms (wheeze, stridor, dyspnoea) and/or cardiovascular collapse (hypotension, tachycardia, syncope). Management: adrenaline 0.5mg IM (adult, mid-outer thigh) IMMEDIATELY - first and most important treatment; call 999/112/911; lay flat with legs elevated (sitting up if breathing difficulty); oxygen; IV fluid; IM/IV chlorphenamine and hydrocortisone (secondary, not first-line). Auto-injector (EpiPen/Jext) for all at risk of anaphylaxis - ALWAYS carry two. Refer to allergy specialist after episode.",
    },
    "hair_disorders": {
        "name": "Hair and Scalp Disorders",
        "alopecia_areata": "Alopecia areata (AA): autoimmune T-cell-mediated attack on anagen hair follicles. Presents as sharply demarcated round/oval patches of non-scarring hair loss, typically on scalp. Exclamation mark hairs (hairs that are thicker distally, broken near the scalp) are characteristic. Classification: AA (patchy), alopecia totalis (AT - total scalp loss), alopecia universalis (AU - total body hair loss). Nail pitting can occur. Spontaneous regrowth in 50% within 1 year for limited disease, but recurrence common. Treatment: intralesional corticosteroids (triamcinolone acetonide 5-10mg/mL injections) most effective for limited disease; topical steroids/minoxidil adjuncts; contact immunotherapy (DPCP/DNCB) for extensive disease; baricitinib (JAK1/2 inhibitor, first FDA/EMA approved systemic for severe AA, 2022) and ritlecitinib (JAK3/TEC family) - significant advances for severe/refractory disease.",
        "androgenetic_alopecia": "Most common cause of hair loss in both men and women. Male pattern baldness (Hamilton-Norwood classification): frontotemporal recession with vertex thinning. Female pattern hair loss (FPHL, Ludwig classification): diffuse thinning over crown with preservation of frontal hairline (broader parting). DHT-mediated miniaturisation of genetically susceptible follicles. Treatment: minoxidil 2/5% solution or 5% foam (applied topically OD/BD to scalp - increases anagen phase, works in both sexes, requires ongoing use); finasteride 1mg OD (5-alpha reductase inhibitor, oral, men only - teratogenic, females of childbearing age must not handle crushed tablets); dutasteride (more potent dual 5-alpha reductase inhibitor, off-label); hair transplantation (FUE/FUT); low-level laser therapy (LLLT) as adjunct.",
        "telogen_effluvium": "Diffuse hair shedding following a physiological or psychological stressor 2-3 months prior (the lag time reflects the telogen phase duration). Triggers: high fever/illness, major surgery, childbirth (postpartum telogen effluvium - very common, self-limiting), crash dieting/nutritional deficiency (iron, ferritin, protein, zinc, vitamin D), emotional stress, thyroid dysfunction, medications (anticoagulants, retinoids, beta-blockers, hormonal changes). Investigation: FBC, ferritin (aim over 70 mcg/L for hair), thyroid function, zinc, vitamin D, LFTs, hormonal profile if indicated. Management: treat underlying cause, nutritional optimisation, reassurance (self-limiting in most cases, regrowth in 6-12 months), topical minoxidil can speed recovery.",
        "scarring_alopecia": "Lichen planopilaris (LPP): perifollicular scaling and erythema, predominantly affects crown and temples in women, leads to permanent scarring. Frontal fibrosing alopecia (FFA): band-like recession of frontal and temporal hairline with perifollicular inflammation, loss of eyebrows/eyelashes, increasingly recognised. Discoid lupus erythematosus (DLE): can cause scarring alopecia. Diagnosis requires scalp biopsy in many cases. Treatment: topical/intralesional corticosteroids, hydroxychloroquine, off-label JAK inhibitors showing promise for LPP/FFA.",
    },
    "nail_disorders": {
        "name": "Nail Disorders",
        "onychomycosis": "Fungal nail infection (tinea unguium) is the most common nail disorder, affecting approximately 10% of the general population (up to 30% over 60 years). Predominantly Trichophyton rubrum. Types: distal lateral subungual onychomycosis (DLSO - most common, nail thickening, yellowing, subungual debris starting at free edge), white superficial (white powdery patches on nail surface), proximal subungual (immunosuppressed), total dystrophic (end-stage). Diagnosis: nail clippings including subungual debris for microscopy (PAS stain) and culture - essential before starting treatment. Treatment: oral terbinafine 250mg OD for 3 months (toenails), 6 weeks (fingernails) - most effective; itraconazole pulse therapy (400mg OD for 1 week per month, 3 pulses); topical amorolfine 5% nail lacquer (for single nail involvement, superficial infection, or as adjunct to systemic) or efinaconazole/ciclopirox nail solutions.",
        "nail_psoriasis": "See psoriasis section. Pitting is pathognomonic of psoriasis. Distinguish from onychomycosis (may coexist) - culture before antifungal treatment.",
        "paronychia": "Acute paronychia: acute bacterial infection of the nail fold, usually Staphylococcus aureus following minor trauma/nail biting. Painful erythematous swelling of the nail fold, may progress to abscess. Treatment: warm soaks, topical antibiotics for mild cases, oral flucloxacillin if spreading, incision and drainage of fluctuant abscess. Chronic paronychia: chronic inflammation of the nail fold, usually multifactorial (Candida, irritant exposure - typically affecting those with repeated wet work: healthcare workers, cleaners, bartenders, cooks). Management: keep dry, barrier protection, topical antifungal+steroid combinations.",
    },
    "pigmentation_disorders": {
        "name": "Pigmentation and Photodermatology",
        "vitiligo": "Vitiligo: acquired autoimmune depigmentation. T-cell mediated destruction of melanocytes. Prevalence approximately 1-2% globally, affects all skin types but visually more apparent and more psychologically distressing in darker skin tones. Types: non-segmental (bilateral, symmetric, progressive - most common), segmental (unilateral, dermatomal distribution, stable after initial phase). Wood's lamp accentuates depigmentation (bright white fluorescence). Associated autoimmune conditions: thyroid disease (most common - check TFTs), type 1 DM, Addison's disease, pernicious anaemia. Treatment: topical calcineurin inhibitors (tacrolimus, pimecrolimus - good for face/sensitive areas), topical steroids, NB-UVB phototherapy (most effective for non-segmental, 2-3x/week for 1-2 years minimum), JAK inhibitors (ruxolitinib 1.5% cream - Opzelura/Jakavi cream, FDA/EMA approved for non-segmental vitiligo, most effective repigmentation therapy), afamelanotide implant (alpha-MSH analogue, increases MC1R signalling - used with NB-UVB in darker skin types).",
        "melasma": "Acquired hypermelanosis, predominantly in women (90%), of childbearing age, in darker skin types (Fitzpatrick III-VI). Symmetrical brown-grey macular hyperpigmentation on sun-exposed areas: face (centro-facial, malar, mandibular patterns), forearms. Exacerbated by UV, hormonal changes (pregnancy - chloasma, oral contraceptives, HRT), heat. Epidermal (responds better) vs dermal (Wood's lamp does not accentuate) vs mixed type. Management: strict sun protection (SPF 50+, reapply every 2 hours, physical block preferred) - ESSENTIAL and the most important intervention; topical treatments: hydroquinone 2-4% (most evidence, limit to 3-6 month courses to avoid ochronosis), triple combination Kligman's formula (hydroquinone 4%+tretinoin 0.05%+fluocinolone 0.01% - Tri-Luma), azelaic acid 15-20%, tranexamic acid (oral 250mg BD or topical 2-5%), kojic acid, niacinamide, vitamin C; chemical peels (glycolic acid, TCA - cautious in dark skin), laser/IPL (risk of PIH in dark skin, use cautiously); tranexamic acid showing good emerging evidence oral or topical.",
        "photodermatology": "Sun protection is the most important preventive measure for skin cancer and photoageing. SPF (Sun Protection Factor) measures UVB protection; broad-spectrum products also protect against UVA (PPD/PA rating). SPF 30 blocks 97% UVB; SPF 50 blocks 98%. Recommended application: 1/4 teaspoon (approximately 1.5g) for face/neck, apply 30 minutes before sun exposure, reapply every 2 hours. Fitzpatrick skin phototypes: I (always burns, never tans - highest cancer risk), II, III, IV, V, VI (rarely burns, darkest skin - lower cancer risk but melanoma can be more aggressive when it occurs, often diagnosed later). Polymorphous light eruption (PLE): most common photodermatosis, itchy papules/vesicles on sun-exposed skin appearing hours after sun exposure, self-limiting. Prophylaxis: sunscreen, NB-UVB desensitisation, hydroxychloroquine.",
    },
    "dermatology_emergency": {
        "name": "Dermatological Emergencies",
        "sjs_ten": "Stevens-Johnson Syndrome (SJS) and Toxic Epidermal Necrolysis (TEN) are severe life-threatening mucocutaneous reactions characterised by epidermal detachment. SJS: under 10% BSA detachment; TEN: over 30%; SJS/TEN overlap: 10-30%. Drug-induced in majority of cases (most common culprits: allopurinol, sulfonamide antibiotics - cotrimoxazole, anticonvulsants - carbamazepine, phenytoin, lamotrigine, NSAIDs - especially oxicams, nevirapine). EMERGENCY: hospitalisation in burns unit/ICU, immediate withdrawal of causative drug, supportive care (IV fluids, temperature regulation, nutritional support, wound care, ophthalmology review for eye involvement), IVIG/ciclosporin controversially used, mortality 10% (SJS) to 30-40% (TEN). SCORTEN severity scoring. HLA-B*5801 (allopurinol-SJS risk in Han Chinese) and HLA-B*1502 (carbamazepine-SJS risk in South-East Asian populations) - pharmacogenomic testing before prescribing in at-risk populations.",
        "dress": "Drug Reaction with Eosinophilia and Systemic Symptoms (DRESS) / Drug-induced Hypersensitivity Syndrome (DiHS). Severe drug reaction with latency 2-8 weeks after initiation. Triad: extensive morbilliform rash (may progress to exfoliative erythroderma), fever, and internal organ involvement (liver most common - elevated LFTs, hepatitis; renal involvement; lung; heart - myocarditis). Peripheral eosinophilia and atypical lymphocytosis. Most commonly caused by: aromatic anticonvulsants (phenytoin, carbamazepine, phenobarbitone), allopurinol, dapsone, abacavir, minocycline, sulfonamides. Hospitalisation required, stop causative drug, systemic corticosteroids in most cases. RegiSCAR scoring criteria for diagnosis.",
        "erythroderma": "Generalised erythroderma (exfoliative dermatitis): 90%+ BSA affected with erythema and scaling. Causes: pre-existing skin disease flare (psoriasis, atopic eczema - most common), drug reaction, cutaneous T-cell lymphoma (CTCL)/Sezary syndrome, pityriasis rubra pilaris, rarer causes. Complications: thermoregulatory failure (heat loss through vasodilated skin), high-output cardiac failure, secondary infection, protein loss, electrolyte imbalance. EMERGENCY: hospital admission, treat underlying cause, emollients, temperature regulation, monitoring of cardiac status, nutritional support.",
        "cellulitis_severe": "Severe cellulitis/necrotising fasciitis - see infections section. Pain disproportionate to appearance, dusky/grey skin colour, rapid spread beyond marked borders, crepitus, systemic sepsis = necrotising fasciitis until proven otherwise. SURGICAL EMERGENCY. Do not delay surgery for imaging.",
        "anaphylaxis_derm": "See urticaria/angioedema section. IM adrenaline 0.5mg immediately, call 112/999/911.",
    },
}

def save_knowledge():
    try:
        with open(DATA_DIR / "derma_knowledge.json", "w", encoding="utf-8") as f:
            json.dump(KNOWLEDGE, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.warning("Khong ghi duoc derma_knowledge.json: %s", e)

def load_sessions():
    sf = DATA_DIR / "sessions.json"
    if sf.exists():
        with open(sf) as f: return json.load(f)
    return {}

def save_session(sid, data):
    sessions = load_sessions()
    sessions[sid] = {**data, "updated": datetime.datetime.now().isoformat()}
    with open(DATA_DIR / "sessions.json", "w") as f: json.dump(sessions, f, indent=2)

def is_online():
    if not REQUESTS_OK: return False
    try: req_lib.get("https://8.8.8.8", timeout=3); return True
    except: return False

def extract_pdf_text(filepath):
    if not FITZ_OK: return "[PDF extraction unavailable]"
    try:
        doc = fitz.open(str(filepath))
        text = "".join(page.get_text() for page in doc)
        doc.close(); return text[:8000]
    except Exception as e: return f"[PDF error: {e}]"

DEFAULT_SYSTEM = (
    "You are DermaCare AI, a dermatology health research assistant. Help patients understand "
    "skin conditions, medications, procedures, and skin health from published dermatology "
    "literature. ALWAYS start with a brief AI research disclaimer. Reference AAD, BAD, EADV, "
    "WHO, NICE, IADVL guidelines. ALWAYS end reminding them to consult a qualified dermatologist. "
    "For dermatological emergencies (SJS/TEN, anaphylaxis, necrotising fasciitis, erythroderma) "
    "advise calling 112/999/911 immediately or attending emergency department. "
    "Be specific about skin of colour considerations (Fitzpatrick types IV-VI) where relevant "
    "as presentations often differ and are under-represented in dermatology training."
)

def call_ai(prompt, system_prompt=None, max_tokens=2500, provider=None, api_key=None):
    if not AI_PROVIDERS_OK: return None, "ai_providers_missing"
    provider = validate_provider(provider)
    effective_key = sanitise_api_key(api_key) or DEFAULT_PROVIDER_KEYS.get(provider, "") or (API_KEY if provider == "anthropic" else "")
    if not effective_key or not REQUESTS_OK or not is_online(): return None, "offline_or_no_key"
    text, mode = ai_providers.call_ai(provider, effective_key, prompt, system_prompt or DEFAULT_SYSTEM, max_tokens)
    if text is None: log.error(f"{provider} API error: {mode}"); return None, mode
    return text, "live_ai"

def build_offline_response(topic, patient_info=None):
    topic_l = topic.lower()
    kb_key = next((k for k in KNOWLEDGE if k.replace("_"," ") in topic_l or topic_l in k.replace("_"," ")), None)
    lines = ["# DermaCare AI Research Report", f"**Topic:** {topic}",
             "**Mode:** Offline Research (Embedded Dermatology Knowledge Base)", "",
             "> DISCLAIMER: AI-generated educational information. NOT medical advice. "
             "ALWAYS consult a qualified dermatologist. "
             "EMERGENCY: Call 112 (India) / 999 (UK) / 911 (US).", "", "---", ""]
    if kb_key:
        kb = KNOWLEDGE[kb_key]
        lines.append(f"## {kb.get('name', topic)}\n")
        for field, value in kb.items():
            if field == "name": continue
            if isinstance(value, str): lines += [f"**{field.replace('_',' ').title()}:** {value}", ""]
    else:
        lines += [f"## Research Overview: {topic}", "",
                  f"Enable live AI in Settings for detailed research on {topic}.", ""]
    lines += ["---", "## India Dermatology Resources",
              "- IADVL: Indian Association of Dermatologists, Venereologists and Leprologists (iadvl.org)",
              "- AIIMS Dermatology, New Delhi: aiims.edu",
              "- NIMHANS / PGI Chandigarh Dermatology",
              "- Apollo / Fortis Dermatology Departments",
              "- Emergency: 112", "", f"WARNING - {DISCLAIMER}"]
    return "\n".join(lines)

@app.route("/")
def index(): return send_from_directory(str(STATIC_DIR), "index.html")

@app.route("/<path:filename>")
def static_files(filename): return send_from_directory(str(STATIC_DIR), filename)

@app.route("/api/health")
def health(): return jsonify({"status":"ok","version":VERSION,"online":is_online(),"pdf_extract":FITZ_OK,"timestamp":datetime.datetime.now().isoformat()})

@app.route("/api/upload", methods=["POST"])
def upload():
    if "files" not in request.files: return jsonify({"error":"No files"}), 400
    session_id = request.form.get("session_id") or str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id; session_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for f in request.files.getlist("files"):
        if not f.filename: continue
        ext = Path(f.filename).suffix.lower()
        safe = f"{uuid.uuid4().hex}{ext}"; dest = session_dir / safe; f.save(str(dest))
        extracted = extract_pdf_text(dest) if ext == ".pdf" else ""
        results.append({"original":f.filename,"saved":safe,"type":"pdf" if ext==".pdf" else ("image" if ext in [".jpg",".jpeg",".png"] else "text"),"size_kb":round(dest.stat().st_size/1024,1),"has_content":bool(extracted)})
    existing = load_sessions().get(session_id, {})
    save_session(session_id, {"session_id":session_id,"files":existing.get("files",[])+results})
    return jsonify({"success":True,"session_id":session_id,"uploaded":len(results),"files":results,"disclaimer":DISCLAIMER})

@app.route("/api/analyse", methods=["POST"])
def analyse():
    data = request.json or {}
    if not rate_limit_check(): return jsonify({"error":"Rate limit exceeded","mode":"rate_limited"}), 429
    topic = data.get("topic","General Dermatology"); condition = data.get("condition","")
    patient_info = data.get("patient_info",{})
    provider = validate_provider(data.get("provider","anthropic"))
    effective_key = sanitise_api_key(data.get("api_key","")) or DEFAULT_PROVIDER_KEYS.get(provider,"") or (API_KEY if provider=="anthropic" else "")
    prompt = f"""Dermatology Health Research Request:
Topic/Condition: {topic} / {condition}
Patient: Age {patient_info.get('age','not specified')}, Skin type: {patient_info.get('skin_type','not specified')}
Symptoms: {patient_info.get('symptoms','not specified')}, Duration: {patient_info.get('duration','not specified')}
Medications: {patient_info.get('medications','none')}, Other conditions: {patient_info.get('conditions','none')}
Cover: clinical overview, differential diagnosis, investigations (dermoscopy, biopsy, patch testing, culture), 
evidence-based treatment (topical/systemic/procedural/biologic), skincare recommendations, 
sun protection, skin of colour considerations (Fitzpatrick types IV-VI), emergency red flags, 
India dermatology resources, questions for dermatologist. 
Reference AAD, BAD, EADV, WHO, NICE, IADVL guidelines."""
    result, mode = call_ai(prompt, provider=provider, api_key=effective_key) if (effective_key and is_online()) else (None,"offline")
    if not result: result = build_offline_response(topic, patient_info); mode = "offline"
    return jsonify({"success":True,"mode":mode,"analysis":result,"topic":topic,"disclaimer":DISCLAIMER,"timestamp":datetime.datetime.now().isoformat()})

@app.route("/api/condition/<condition_name>")
def condition_detail(condition_name):
    cn = condition_name.lower().replace("-","_").replace(" ","_")
    if cn in KNOWLEDGE: return jsonify({"success":True,"mode":"offline_kb","condition":KNOWLEDGE[cn],"disclaimer":DISCLAIMER})
    provider = validate_provider(request.args.get("provider","anthropic"))
    effective_key = sanitise_api_key(request.args.get("api_key","")) or DEFAULT_PROVIDER_KEYS.get(provider,"") or (API_KEY if provider=="anthropic" else "")
    prompt = f"Comprehensive dermatology research on {condition_name}: definition, prevalence, pathophysiology, clinical features (including in darker skin types), diagnosis, differential diagnosis, evidence-based treatment, prognosis. Reference AAD, BAD, EADV, NICE, IADVL."
    result, mode = call_ai(prompt, provider=provider, api_key=effective_key)
    if not result: result = build_offline_response(condition_name); mode = "offline"
    return jsonify({"success":True,"mode":mode,"content":result,"disclaimer":DISCLAIMER})

@app.route("/api/skin/assess", methods=["POST"])
def assess_skin():
    data = request.json or {}
    skin_type = data.get("skin_type",""); concerns = data.get("concerns","")
    age = data.get("age",""); allergies = data.get("allergies","")
    provider = validate_provider(data.get("provider","anthropic"))
    effective_key = sanitise_api_key(data.get("api_key","")) or DEFAULT_PROVIDER_KEYS.get(provider,"") or (API_KEY if provider=="anthropic" else "")
    prompt = f"""Personalised Skincare Research:
Fitzpatrick Skin Type: {skin_type}, Age: {age}
Primary Concerns: {concerns}, Known Allergies/Sensitivities: {allergies}
Please provide evidence-based skincare research covering:
1. Appropriate cleanser type and frequency
2. Moisturiser/emollient recommendations
3. Sun protection (SPF type, application)
4. Active ingredients relevant to stated concerns (with evidence level)
5. Ingredients to avoid for stated skin type/concerns
6. Morning vs evening routine structure
7. Specific considerations for skin of colour if applicable
Note: This is general skincare research, not a prescription. Consult a dermatologist for medical conditions."""
    result, mode = call_ai(prompt, provider=provider, api_key=effective_key)
    if not result:
        result = f"Personalised skincare research for {skin_type} skin with {concerns} concerns. Enable live AI in Settings for detailed research. General advice: gentle cleanser, broad-spectrum SPF 30+ daily, moisturiser appropriate to skin type. Consult your dermatologist for personalised recommendations."
        mode = "offline"
    return jsonify({"success":True,"mode":mode,"content":result,"disclaimer":DISCLAIMER})

@app.route("/api/chat/send", methods=["POST"])
def chat_send():
    data = request.json or {}
    message = data.get("message","").strip()
    if not message: return jsonify({"error":"Empty message"}), 400
    provider = validate_provider(data.get("provider","anthropic"))
    effective_key = sanitise_api_key(data.get("api_key","")) or DEFAULT_PROVIDER_KEYS.get(provider,"") or (API_KEY if provider=="anthropic" else "")
    result = None
    if data.get("request_ai") and is_online() and effective_key:
        result, _ = call_ai(f"Dermatology patient question: '{message}'. Respond in 3-4 paragraphs, compassionate and evidence-based. Include skin of colour considerations where relevant. End with: consult dermatologist reminder and for emergencies (SJS/TEN, anaphylaxis, necrotising fasciitis) call 112/999/911 immediately.", max_tokens=800, provider=provider, api_key=effective_key)
    return jsonify({"success":True,"ai_response":result,"disclaimer":"Not medical advice. Consult your dermatologist."})

@app.route("/api/report/generate", methods=["POST"])
def generate_report():
    data = request.json or {}
    topic = data.get("topic","General Dermatology"); patient = data.get("patient_info",{})
    provider = validate_provider(data.get("provider","anthropic"))
    effective_key = sanitise_api_key(data.get("api_key","")) or DEFAULT_PROVIDER_KEYS.get(provider,"") or (API_KEY if provider=="anthropic" else "")
    content = build_offline_response(topic, patient)
    if effective_key and is_online():
        ai_content, _ = call_ai(f"Generate comprehensive dermatology research report for: {topic}. Patient: {patient}. Cover diagnosis, treatment options (topical/systemic/biologic/procedural), skincare, sun protection, follow-up.", max_tokens=3500, provider=provider, api_key=effective_key)
        if ai_content: content = ai_content
    report_id = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    report = {"report_id":report_id,"generated":datetime.datetime.now().isoformat(),"topic":topic,"patient":patient,"content":content,"disclaimer":DISCLAIMER}
    with open(REPORTS_DIR / f"{report_id}.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2,ensure_ascii=False)
    return jsonify(report)

@app.route("/api/providers")
def list_providers():
    if not AI_PROVIDERS_OK: return jsonify({"providers":[],"error":"ai_providers module not available"})
    return jsonify({"providers":[{"id":k,"label":v["label"],"default_model":v["default_model"],"key_prefix":v["key_prefix"],"get_key_url":v["get_key_url"],"server_default_configured":bool(DEFAULT_PROVIDER_KEYS.get(k))} for k,v in ai_providers.PROVIDERS.items()],"online":is_online()})

@app.route("/api/status")
def status():
    any_key = bool(API_KEY) or any(DEFAULT_PROVIDER_KEYS.values())
    return jsonify({"server":"running","version":VERSION,"online":is_online(),"mode":"live_ai" if (any_key and is_online()) else "offline_research","capabilities":{"pdf":FITZ_OK,"images":PIL_OK,"live_ai":bool(any_key and is_online()),"offline":True,"multi_provider":AI_PROVIDERS_OK,"rate_limiting":True,"aes256_frontend":True},"knowledge_base":list(KNOWLEDGE.keys()),"providers":list(ai_providers.PROVIDERS.keys()) if AI_PROVIDERS_OK else [],"disclaimer":DISCLAIMER})



@app.route("/api/resolve", methods=["POST"])
def resolve_multi_ai():
    """Ambiguity Resolver: query multiple providers in parallel, synthesise best answer."""
    data = request.json or {}
    if not rate_limit_check():
        return jsonify({"error": "Rate limit exceeded"}), 429
    prompt   = str(data.get("prompt", "")).strip()[:4000]
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    pairs_raw = data.get("providers", [])
    if not isinstance(pairs_raw, list) or len(pairs_raw) < 1:
        return jsonify({"error": "No providers specified"}), 400
    if not AI_PROVIDERS_OK:
        return jsonify({"error": "ai_providers module not available"}), 500

    pairs = []
    for p in pairs_raw[:6]:
        pid = validate_provider(p.get("provider", ""))
        key = sanitise_api_key(p.get("key", ""))
        if pid and key:
            pairs.append((pid, key))
    if not pairs:
        return jsonify({"error": "No valid provider+key pairs"}), 400

    results = ai_providers.call_multi_ai(pairs, prompt, DEFAULT_SYSTEM, 1500)

    # Build synthesis prompt from successful responses
    successes = [r for r in results if r.get("success") and r.get("text")]
    synthesis = None
    if len(successes) >= 2:
        synth_parts = []
        for r in successes:
            label = r.get("label", r.get("provider", "AI"))
            text  = (r.get("text") or "")[:1200]
            synth_parts.append(f"=== {label} ===\n{text}")
        synth_prompt = (
            "You are a medical research synthesis engine. "
            "Below are responses from multiple AI systems to the same medical research question.\n"
            "Question: " + prompt + "\n\n"
            + "\n\n".join(synth_parts) + "\n\n"
            "Synthesise the best, most complete, evidence-based answer. "
            "Note any disagreements between the AIs. "
            "Start with the most important finding. "
            "This is for research/education only — remind the user to consult a qualified clinician."
        )
        # Use first successful provider to synthesise
        synth_provider = successes[0]["provider"]
        synth_key = None
        for orig_p, orig_k in pairs:
            if orig_p == synth_provider:
                synth_key = orig_k
                break
        if synth_key:
            synth_text, _ = ai_providers.call_ai(
                synth_provider, synth_key, synth_prompt,
                "You are a medical research synthesis assistant.", 2000
            )
            synthesis = synth_text

    return jsonify({
        "success": True,
        "responses": results,
        "synthesis": synthesis,
        "disclaimer": DISCLAIMER,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", default=os.environ.get("DERMACARE_HOST", "127.0.0.1"), help="mac dinh 127.0.0.1 (chi may nay); dung 0.0.0.0 neu muon mo cho mang noi bo")
    args = parser.parse_args()
    save_knowledge()
    log.info("=" * 60); log.info(f"  DermaCare AI Server v{VERSION} - Port {args.port}")
    log.info(f"  Online: {is_online()}"); log.info(f"  URL: http://{args.host}:{args.port}"); log.info("=" * 60)
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)

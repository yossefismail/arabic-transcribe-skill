# تفريغ عربي

**العربية** · [English](README.md)

مهارة وكيل تحوّل تسجيلًا صوتيًا عربيًا — رسالة واتساب صوتية، أو اجتماعًا، أو مقابلة —
إلى **صفحة HTML واحدة مكتفية بذاتها**: مشغّل صوت مدمج، وتفريغ ملوّن حسب المتحدث
يتظلّل أثناء التشغيل، ونص ينقلك النقر عليه إلى موضعه في التسجيل، وملخص مكتوب، وقائمة
بنود تنفيذ — بتنسيق «لوحة الرسم الهندسي» مع شريط للحظات المفتاحية على خط الزمن.

مبنية للعربية باللهجة المصرية مع تبديل إلى الإنجليزية، وتتعامل مع الفصحى وبقية اللهجات
أيضًا.

**كل شيء يعمل على جهازك.** لا خدمات سحابية، ولا رفع، ولا حساب، ولا مفتاح — الصوت
والنماذج والصفحة النهائية تبقى على القرص. حركة الشبكة الوحيدة على الإطلاق هي تنزيل
النماذج في المرة الأولى.

```sh
npx skills add yossefismail/arabic-transcribe-skill
```

![سير عمل التفريغ العربي: ثماني مراحل من تسجيل صوتي إلى لوحة جاهزة](assets/workflow-ar.png?v=2)

## ما الذي تحصل عليه

| المخرج | ما هو |
|---|---|
| ‎`<name> - arabic transcript.html` | المخرج الأساسي. الصوت مضمَّن بترميز base64، فيعمل دون اتصال ويمكن إرساله كملف واحد. ألوان للمتحدثين، وتظليل للجملة الجارية، ونقر للانتقال، وبطاقة ملخص بمربعات اختيار تحفظ حالتها. |
| ‎`<name> - arabic transcript.txt` | نسخة نصية احتياطية بعلامات RLM/LRM حتى يظهر العربي والإنجليزي معًا بشكل صحيح في Notepad. |
| ‎`<name> - enhanced audio.m4a` | الصوت بعد التنقية، منفردًا. |

## النموذج والدقة

التفريغ عبر [‎`IFM/Jais-ASR`](https://huggingface.co/IFM/Jais-ASR) — 5.72 مليار معامل،
رخصة Apache-2.0، ويُحمَّل بدقة 4 بت. معدلات الخطأ المنشورة على [لوحة التعرّف العربي
المفتوحة](https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard)
(الأقل أفضل، وهي متوسط ست مجموعات):

| مجموعة الاختبار | WER % | CER % |
|---|---:|---:|
| **المتوسط** | **22.85** | **8.37** |
| Common Voice 18 — فصحى مقروءة | 8.56 | 2.70 |
| MGB-2 — فصحى، أخبار إذاعية | 9.59 | 3.92 |
| MASC نظيف — فصحى، محاضرات | 16.12 | 3.76 |
| MASC مشوّش — فصحى، محاضرات | 25.44 | 7.51 |
| SADA — لهجة سعودية، بث | 30.85 | 14.33 |
| Casablanca — ثماني لهجات، حوار | 46.56 | 18.02 |

هذا أفضل متوسط على اللوحة، لكنه ليس الأفضل في كل مجموعة. للمقارنة:

| النموذج | متوسط WER % |
|---|---:|
| **Jais-ASR** | **22.85** |
| Audar-ASR-V1-Turbo | 23.17 |
| Speechmatics Melia 1 | 23.51 |
| ElevenLabs Scribe v2 | 24.54 |
| Cohere transcribe-arabic | 25.87 |
| omniASR-LLM-7B | 28.32 |

الحوار متعدد اللهجات هو الحالة الصعبة: احسب خطأً يقارب خمسة أضعاف الخطأ في فصحى
مقروءة نظيفة. والأرقام أعلاه بقياس bf16، بينما تعمل المهارة بدقة 4 بت، فوقع الخطأ
فيها أعلى قليلًا.

**التغطية:** الفصحى الحديثة، و19 لهجة: ‎`DZ` جزائرية، و‎`BH` بحرينية، و‎`EG` مصرية،
و‎`AE` إماراتية، و‎`IQ` عراقية، و‎`JO` أردنية، و‎`KW` كويتية، و‎`LB` لبنانية، و‎`LY` ليبية،
و‎`MR` موريتانية، و‎`MA` مغربية، و‎`OM` عُمانية، و‎`PS` فلسطينية، و‎`QA` قطرية، و‎`SA`
سعودية، و‎`SD` سودانية، و‎`SY` سورية، و‎`TN` تونسية، و‎`YE` يمنية. ويضاف إليها التبديل
إلى الإنجليزية والفرنسية والإسبانية والإيطالية، وتبقى المقاطع الأجنبية بالحرف اللاتيني.

الخيار ‎`--language auto` يجعل النموذج يكتشف اللهجة، و‎`--language EG`
(الافتراضي) يثبّتها.

التفاصيل الكاملة والمنهجية والاقتباس في [references/model.md](references/model.md).

## المتطلبات

• **Python 3.12** — الأمر ‎`py -3.12` على ويندوز، و‎`python3.12` في غيره.

• **بطاقة NVIDIA تدعم CUDA** بذاكرة 6 غيغابايت أو أكثر. المهارة مُختبَرة على RTX 4050
بذاكرة 6 غيغابايت، وميزانية توزيع الطبقات تُحسب من الذاكرة الحرة وقت التحميل، ويمكن
تجاوزها بالمتغيّرين ‎`JAIS_GPU_MEM` و‎`JAIS_CPU_MEM`.

• **نحو 12 غيغابايت مساحة حرة** للنموذج المكمَّم، إضافة إلى ذاكرة Hugging Face
المؤقتة.

• **دون حساب أو مفتاح.** تمييز المتحدثين يعمل بنموذج NVIDIA Nemotron-3-Diarization
عبر أداة ‎`nemo-speech`، وهي تنزّل نفسها عند أول استخدام (نحو 110 ميغابايت، متحقَّق
منها بـ SHA-256).

ويندوز هو المنصة المُختبَرة. من المفترض أن يعمل macOS ولينكس، لكنهما غير مُختبَرين.

## التثبيت

أمر واحد عبر [واجهة skills](https://github.com/vercel-labs/skills):

```sh
npx skills add yossefismail/arabic-transcribe-skill
```

تكتشف المهارة الوكلاء المثبّتين عندك وتضعها في المكان الصحيح. بلا خيارات تُثبَّت على
مستوى المشروع، ومع ‎`-g` في مجلد المستخدم:

| الوكيل | مسار المستخدم |
|---|---|
| OpenCode | ‎`~/.config/opencode/skills/arabic-transcribe` |
| Claude Code | ‎`~/.claude/skills/arabic-transcribe` |

والتثبيت على مستوى المشروع يُسجَّل في ‎`skills-lock.json` مع المصدر وبصمة المحتوى.

```sh
npx skills add yossefismail/arabic-transcribe-skill -g -a opencode
npx skills add yossefismail/arabic-transcribe-skill -g -a claude-code
npx skills list
npx skills update
npx skills remove arabic-transcribe
```

المستودع يحتوي مهارة واحدة، فلا حاجة إلى العلم ‎`--skill`.

تفضّل نسخة يدوية؟ استنسخ المستودع حيث يبحث وكيلك عن المهارات:

```sh
git clone https://github.com/yossefismail/arabic-transcribe-skill.git \
  ~/.config/opencode/skills/arabic-transcribe
```

على ويندوز مسار OpenCode هو ‎`C:\Users\<you>\.config\opencode\skills\`.

ثم ثبّت اعتماديات Python مرة واحدة. الشرح الكامل في
[references/setup.md](references/setup.md)، وفيه أيضًا حلقة curl للإنقاذ إذا توقّف
تنزيل النموذج:

```sh
py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cu128
py -3.12 -m pip install torchaudio --index-url https://download.pytorch.org/whl/cu128
py -3.12 -m pip install transformers==5.16.1 bitsandbytes soundfile imageio-ffmpeg huggingface-hub noisereduce
py -3.12 -m pip install --no-deps resemble-enhance tabulate
```

لا شيء غير ذلك: أداة التمييز تُنزَّل تلقائيًا عند أول استخدام.

وللتحقق يجب أن يطبع الأمر التالي إصدارًا ينتهي بـ ‎`+cu128` ثم ‎`True`:

```sh
py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## الاستخدام

اطلب من وكيلك تفريغ تسجيل، أو شغّل السكربتات الثلاثة بنفسك:

```sh
# 1. transcribe, diarize, build the player
py -3.12 scripts/run_pipeline.py "voice note.m4a" --outdir out

# 2. write the summary (agent step) and inject it into both files
py -3.12 scripts/add_summary.py "out/voice note" --summary-file summary.md

# 3. apply the house UI and the key-moment ticks
py -3.12 scripts/apply_ui.py "out/voice note" \
  --title "تفريغ|المكالمة" --project "MEETING NOTES" \
  --date "01·11·2026" --kicker "RECORDING 04 — TRANSCRIPT INDEX"
```

‎`SKILL.md` هو الدليل التشغيلي الكامل الذي يتبعه الوكيل: كيف يشغّل المسار، وماذا يحتوي
الملخص الجيد، وقواعد واجهة المهارة.

أعلام مفيدة: ‎`--quant 8bit` و‎`--no-enhance`
و‎`--max-seconds N` و‎`--language EG|auto` مع رمز اللهجة.

## محلي بالكامل

• **لا شيء يغادر الجهاز.** الصوت يبقى على القرص، ولا تتبّع ولا خطوة رفع في أي سكربت.

• **حركة الشبكة محدودة بالتنزيل الأول.** أوزان Jais-ASR من Hugging Face Hub، وأداة
تمييز المتحدثين من NVIDIA: ملف ‎`nemo-speech` التنفيذي (نحو 8 ميغابايت) والنموذج
(102 ميغابايت). لا حساب ولا مفتاح ولا تسجيل لنموذج مقيَّد في أي خطوة.

• **المشغّل ملف واحد.** الصوت مضمَّن فيه بترميز base64 والخطوط مدمجة، فيُفتح على جهاز
لم يرَ هذا المستودع قط ولا يملك اتصالًا بالشبكة.

• **الأداة التنفيذية تُتحقَّق قبل التشغيل.** إصدار ‎`nemo-speech` مثبَّت على v0.3.0
ويُقارَن بـ SHA-256 المنشور قبل التنفيذ. رخصة الأداة Apache-2.0، ورخصة النموذج
OpenMDW-1.1. وللحذف: ‎`~/.cache/arabic-transcribe/nemo-speech/`.

• **إيداع مُدقَّق للنموذج.** ‎`IFM/Jais-ASR` يأتي بشيفرة نمذجة مخصّصة ينفّذها
‎`transformers` عبر ‎`trust_remote_code`، لذا يثبّت المسار الإيداع ‎`JAIS_REVISION`.
اضبط ‎`JAIS_REVISION=main` إن أردت متابعة المستودع.

• **بيئة Python لا تتغيّر.** الـ stub الذي يحتاجه Resemble Enhance يُكتب في مجلد مؤقت
ويُضاف إلى ‎`sys.path`.

• **الصوت داخل الملف.** صفحة HTML الناتجة تضمّن الصوت كاملًا، فمشاركة الملف تعني
مشاركة الصوت.

## استكشاف الأخطاء

انظر [references/setup.md](references/setup.md): تنزيل النموذج الذي يتوقّف، وخطأ
‎`paging file too small`، وخطأ خريطة الأجهزة في bitsandbytes، ومشكلات التمييز في
‎`nemo-speech`.

## البنية

| الملف | الوظيفة |
|---|---|
| ‎`SKILL.md` | تعليمات الوكيل: سير العمل، ومواصفات الملخص، وقواعد الواجهة |
| ‎`scripts/run_pipeline.py` | فك الترميز، ثم التحسين، ثم التمييز، ثم التفريغ، ثم إخراج HTML وTXT |
| ‎`scripts/add_summary.py` | حقن الملخص وقائمة المهام التي يكتبها الوكيل |
| ‎`scripts/apply_ui.py` | تنسيق اللوحة، والمشغّل، وشريط اللحظات المفتاحية |
| ‎`references/setup.md` | إعداد البيئة والنماذج، وجدول الأخطاء |
| ‎`references/model.md` | معدلات خطأ Jais-ASR، ورموز اللهجات، وتبديل اللغات |
| ‎`references/ui-system.md` | النظام البصري والرموز وقواعد التخطيط |
| ‎`assets/fonts/` | خطوط Archivo وJetBrains Mono وCairo (رخصة OFL، انظر OFL.txt) |
| ‎`assets/workflow.png` | مخطط سير العمل بالإنجليزية |
| ‎`assets/workflow-ar.png` | مخطط سير العمل بالعربية |

## الترخيص

الشيفرة والتوثيق برخصة MIT، انظر [LICENSE](LICENSE). والخطوط المرفقة برخصة
SIL Open Font License 1.1، انظر [assets/fonts/OFL.txt](assets/fonts/OFL.txt).
أوزان Jais-ASR برخصة Apache-2.0، وأداة ‎`nemo-speech` برخصة Apache-2.0، وأوزان
Nemotron-3-Diarization برخصة OpenMDW-1.1.

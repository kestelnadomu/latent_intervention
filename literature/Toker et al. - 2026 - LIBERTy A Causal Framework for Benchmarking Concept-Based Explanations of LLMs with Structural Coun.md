---
source_pdf: Toker et al. - 2026 - LIBERTy A Causal Framework for Benchmarking Concept-Based Explanations of LLMs with Structural Coun.pdf
source_pdf_sha256: d3a52cda33505785dda5f3d3b057ebb55127c129900c52185b00d0ee1ef644b5
converted_at_utc: 2026-08-10T09:08:05+00:00
page_count: 39
text_extraction: pdftotext -layout -enc UTF-8
visual_preservation: per-page PNG renders at 200 DPI
---

# Toker et al. - 2026 - LIBERTy A Causal Framework for Benchmarking Concept-Based Explanations of LLMs with Structural Coun

## Source

- PDF: [Toker et al. - 2026 - LIBERTy A Causal Framework for Benchmarking Concept-Based Explanations of LLMs with Structural Coun.pdf](<Toker et al. - 2026 - LIBERTy A Causal Framework for Benchmarking Concept-Based Explanations of LLMs with Structural Coun.pdf>)
- Page image assets: `_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/`
- Method: each page includes extracted layout-preserving text plus a lossless PNG render of the page.
- Use the PNG render whenever extracted text omits or distorts equations, figures, tables, symbols, or page layout.

## PDF Metadata

```text
Title:           LIBERTy: A Causal Framework for Benchmarking Concept-Based Explanations of LLMs with Structural Counterfactuals
Author:          Gilat Toker; Nitay Calderon; Ohad Amosy; Roi Reichart
Creator:         arXiv GenPDF (tex2pdf:57610bf)
Producer:        pikepdf 8.15.1
Custom Metadata: yes
Metadata Stream: yes
Tagged:          no
UserProperties:  no
Suspects:        no
Form:            none
JavaScript:      no
Pages:           39
Encrypted:       no
Page size:       595.276 x 841.89 pts (A4)
Page rot:        0
File size:       1585378 bytes
Optimized:       no
PDF version:     1.7
```

## Embedded Files

```text
0 embedded files
```

## Conversion Diagnostics

```text
[pdftoppm stderr]
Syntax Error: No display font for 'Symbol'
Syntax Error: No display font for 'ArialUnicode'
```

## Pages

### Page 1

![Rendered page 1](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-001.png>)

#### Extracted Text

```text
                                                 LIBERTy: A Causal Framework for Benchmarking Concept-Based
                                                     Explanations of LLMs with Structural Counterfactuals

                                                             Gilat Toker* , Nitay Calderon* , Ohad Amosy, Roi Reichart
                                                   Faculty of Data and Decision Sciences, Technion – Israel Institute of Technology
                                                 {gilatt, nitay}@campus.technion.ac.il, roiri@technion.ac.il
                                                              *
                                                                Second author supervised the project and led the writing.


                                                                   Abstract                             The decisions of these opaque systems are diffi-
                                                                                                        cult to explain, making explainability a central
                                                 Concept-based explanations quantify how high-
arXiv:2601.10700v2 [cs.CL] 18 Jan 2026




                                                 level concepts (e.g., gender or experience) in-        research challenge (Guidotti et al., 2018; Balkir
                                                 fluence model behavior, which is crucial for           et al., 2022; Luo et al., 2024). Among the many
                                                 decision-makers in high-stakes domains. Re-            approaches to explainability, concept-based meth-
                                                 cent work evaluates the faithfulness of such           ods are particularly relevant when the stakeholders
                                                 explanations by comparing them to reference            are decision-makers and end-users (Calderon and
                                                 causal effects estimated from counterfactuals.         Reichart, 2024). These methods focus on quantify-
                                                 In practice, existing benchmarks rely on costly        ing the influence of high-level, human-interpretable
                                                 human-written counterfactuals that serve as
                                                 an imperfect proxy. To address this, we in-
                                                                                                        concepts, such as gender, race, or professional ex-
                                                 troduce a framework for constructing datasets          perience, on model predictions (Kim et al., 2018;
                                                 containing structural counterfactual pairs: LIB-       Künzel et al., 2019; Yeh et al., 2020; Feder et al.,
                                                 ERTy (LLM-based Interventional Benchmark               2021; Wu et al., 2022; Gat et al., 2023).
                                                 for Explainability with Reference Targets).               Recent studies emphasize that explanations lack-
                                                 LIBERTy is grounded in explicitly defined              ing a causal basis often fail to achieve true faithful-
                                                 Structured Causal Models (SCMs) of the text            ness (Lyu et al., 2022; Gat et al., 2023; Yeo et al.,
                                                 generation, interventions on a concept prop-
                                                                                                        2024). In causality, a causal graph encodes con-
                                                 agate through the SCM until an LLM gener-
                                                 ates the counterfactual. We introduce three            cepts as variables and their relationships as edges
                                                 datasets (disease detection, CV screening, and         (Pearl, 2009). This structure enables us to identify
                                                 workplace violence prediction) together with a         the roles of concepts, such as confounders, me-
                                                 new evaluation metric, order-faithfulness. Us-         diators, and colliders, and to estimate the causal
                                                 ing them, we evaluate a wide range of meth-            effect of a target concept on the model (Abraham
                                                 ods across five models and identify substan-           et al., 2022). Despite progress at the intersection
                                                 tial headroom for improving concept-based ex-          of AI and causality (Wood-Doughty et al., 2018;
                                                 planations. LIBERTy also enables systematic
                                                 analysis of model sensitivity to interventions:
                                                                                                        Feder et al., 2021; Wu et al., 2023; Zhang et al.,
                                                 we find that proprietary LLMs show markedly            2023), a fundamental challenge remains: evalu-
                                                 reduced sensitivity to demographic concepts,           ating whether an explanation is faithful requires
                                                 likely due to post-training mitigation. Overall,       comparing it to the true underlying causal mecha-
                                                 LIBERTy provides a much-needed benchmark               nisms. In practice, the ground-truth mechanism is
                                                 for developing faithful explainability methods.1       inaccessible, leaving us without a reliable bench-
                                                                                                        mark for explainability methods.
                                         1       Introduction
                                                                                                           One approach to address this benchmarking chal-
                                         AI systems, especially Large Language Models                   lenge was introduced by Abraham et al. (2022).
                                         (LLMs), increasingly drive decisions in sensitive              They propose using an interventional dataset as
                                         and high-stakes domains where textual input plays              a systematic framework for evaluation of expla-
                                         a central role, such as finance, education, health-            nations. In their interventional dataset, CEBaB,
                                         care, and law (Guidotti et al., 2018; Balkir et al.,           each test example is paired with a human-written
                                         2022; Kasneci et al., 2023; Shui et al., 2023; Luo             counterfactual generated by modifying a concept.
                                         et al., 2024; Nie et al., 2024; Benkirane et al., 2024).       The individual causal concept effect (ICaCE) is
                                             1
                                            https://github.com/GilatToker/                              then estimated by contrasting the model’s outputs
                                         Liberty-benchmark                                              on the original text and its counterfactual. Expla-

                                                                                                    1
```

### Page 2

![Rendered page 2](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-002.png>)

#### Extracted Text

```text
                                                                                    Evaluation Pipeline of Explanation Methods

                                                                                                                                   Concept
                              f is                                                           Explanation Method
                                     tra                                                                                       Importance Score
                                         ine
                                             d   to
                                                    p   red
                                                              ict
                                                                    Y



                                                                                                                                           Com pare


                                                                         Original text

                                                                        Counterfactual
                                                                                                                                REFERENCE
                                                                                                Explained                      Causal Concept
                         LLM
                                                                                                 Model                             Effect
                     (temperature=0)




    Causal Graph of the Text Generation Process

Figure 1: Illustration of LIBERTy: The goal is to evaluate an explanation method Mf that explains the impact
                                →
of changing a concept C (by c ) on model f . Left: The causal graph representing the text generation process.
Exogenous noise variables are denoted by ε, while the endogenous variables (in this illustration) are the concepts
A, B, C, Y , the LLM-generated text xε , and the model prediction f (xε ). The process of generating a structural
                                →
counterfactual for the change c is highlighted in red: C is assigned a new value and propagated through the causal
                                                                 →                                   →
graph (with ε fixed) until the LLM generates the counterfactual xεc . Right: The explanation Mf (xε , c ) is compared
                                                                                                                                             →
against the refrence individual causal concept effect (ICaCE), defined as the difference between f (xεc ) and f (xε ).

nation methods are evaluated by comparing their                                              plated real-world text and author persona, which
importance score (of the concept) against the es-                                            act as exogenous noise variables in the SCM. Coun-
timated causal effect. While CEBaB represents a                                              terfactuals are generated by intervening on a con-
significant step toward causal evaluation, it remains                                        cept (assigning it a new value) and propagating this
limited, especially given LLMs’ current capabili-                                            change through the SCM until the LLM produces
ties. First, CEBaB is confined to sentiment analysis                                         the corresponding counterfactual. As a result, LIB-
of restaurant reviews, which are short, simple texts.                                        ERTy provides structural counterfactuals2 , elimi-
Second, its causal graph comprises only four con-                                            nating the need for costly human annotations and
cepts, with simple relationships (no hierarchical                                            ensuring alignment between the evaluation refer-
structure). Finally, the counterfactuals are written                                         ence target and the DGP.
by human annotators rather than arising from ac-                                                LIBERTy comprises three datasets, each de-
tual interventions in the data-generating process                                            signed around a major societal challenge: disease
(DGP). Consequently, the causal effect references                                            detection, CV screening, and workplace violence
used as “ground truth” for evaluation are them-                                              prediction. We also propose a new evaluation mea-
selves approximations of some unobserved effects.                                            sure, order-faithfulness, that quantifies how well an
                                                                                             explanation method captures the relative ordering
   In this work, we address these limitations by                                             of effects induced by concept interventions. This
introducing a novel framework for generating in-                                             makes it suitable for evaluating explanation meth-
terventional datasets with structural counterfac-                                            ods that provide importance scores on arbitrary
tuals that define reference causal effects: LIB-                                             scales, rather than direct causal effect estimates.
ERTy (LLM-based Interventional Benchmark for                                                    Using LIBERTy, we conduct extensive exper-
Explainability with Reference Targets). LIBERTy,                                             iments to explain five NLP models and LLMs.
illustrated in Figure 1, is based on a simple yet ef-
                                                                                                 2
fective idea: explicitly defining a structured causal                                             Formally, LIBERTy counterfactuals are gold with respect
                                                                                             to the DGP (which the LLM that generates text is part of):
model (SCM) for text generation. In this frame-                                              given the SCM and observed exogenous values, they are gen-
work, the LLM is a component of the SCM that                                                 erated via Pearl’s three-step procedure (Pearl, 2013). However,
instantiates concepts as natural language text. To                                           since the DGP itself and the resulting texts are synthetic, we
                                                                                             use the term silver to refer to these structural counterfactuals.
make LLM outputs more diverse and realistic, we                                              Yet, as LLMs generate an increasing share of real-world data,
provide it with grounding context, such as tem-                                              such setups are both common and practically meaningful.


                                                                                         2
```

### Page 3

![Rendered page 3](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-003.png>)

#### Extracted Text

```text
We benchmark a range of concept-based explana-              explaining human preferences, reward models, and
tion methods, including linear erasure, counterfac-         LLM-as-Judges (Calderon et al., 2025), and detect-
tual generation, matching, and concept attributions.        ing dementia (Peled-Cohen et al., 2025).
Our results show that matching methods based on                Among the most prominent approaches of
representations from a dataset-specific fine-tuned          concept-based explainability, there are Attribution
model perform best overall. Still, we find substan-         methods (Ribeiro et al., 2016; Lundberg and Lee,
tial headroom for improvement, highlighting the             2017; Kim et al., 2018; Yeh et al., 2020), Con-
need for continued explainability research.                 cept Erasure methods (Ravfogel et al., 2022; Bel-
   Besides evaluating explanations, LIBERTy en-             rose et al., 2023), Counterfactual Generation meth-
ables us to analyze the sensitivity of each explained       ods, (Feder et al., 2021; Robeer et al., 2021; Wu
model to concept interventions. For example, when           et al., 2021; Gat et al., 2023), and Matching meth-
the model predicts a candidate’s qualification based        ods, (Veitch et al., 2020; Zhang et al., 2023; Gat
on their CV, we examine how its prediction changes          et al., 2023; Jiang et al., 2025), and Concept Bottle-
when we intervene on the candidate’s race, and              neck models (Koh et al., 2020; Dalvi et al., 2022;
compare this change to the effect specified in the          Yu et al., 2024). Using LIBERTy, we evaluate rep-
SCM. Our results show that fine-tuned models can            resentative methods from those approaches. Never-
track the ground-truth effects of the data. In con-         theless, despite the advantages of concept-based ex-
trast, some LLMs (like GPT-4o) exhibit very low             plainability, particularly for end-users and decision-
sensitivity to demographic concepts, potentially            makers, it remains underexplored relative to token-
due to dedicated post-training alignment.                   level approaches (Calderon and Reichart, 2024).
   Overall, our study represents an important step          A possible reason for this gap is the current lack
toward addressing the long-standing challenge of            of benchmarks that enable rigorous evaluation and
explainability evaluation. By introducing LIB-              systematic comparison.
ERTy, we provide researchers with a reliable, scal-
able, and flexible causal framework for benchmark           Explainability Benchmarks Benchmarking ex-
generation, paving the way for the development of           planations is a highly challenging task, primar-
more faithful explainability methods.                       ily because ground-truth explanations are rarely
                                                            available in real-world datasets (Yang et al., 2019;
2   Related Work                                            Hedström et al., 2023; Lee et al., 2025; Seth and
                                                            Sankarapu, 2025). Most prior evaluation meth-
Concept-based Explainability Concept-based                  ods relied on indirect proxies, such as checking
explainability encompasses methods that quantify            whether different methods agree with one another
the extent to which high-level, human-interpretable         or whether their outputs align with simple heuris-
concepts (features, attributes, variables, rubrics)         tics (Hase and Bansal, 2020; Samek et al., 2021).
that can be explicitly or implicitly conveyed in the        Furthermore, most explainability evaluations have
text influence model predictions. This is in con-           focused on token-level explanations rather than rea-
trast to token-level explanations, which emphasize          soning over high-level semantic concepts (Thorne
tokens through techniques such as attribution or            et al., 2019; Wang et al., 2022; Gurrapu et al.,
attention scores (Calderon and Reichart, 2024; Luo          2023). As mentioned earlier, CEBaB (Concept
et al., 2024; Zhao et al., 2024). Concept-based             Effect Benchmark for NLP, Abraham et al. (2022))
explanations naturally align with human cognitive           was the first dataset to evaluate explainability meth-
processes (Alqaraawi et al., 2020; Kim et al., 2022;        ods under controlled interventions. CEBaB re-
Poeta et al., 2023) and simplify the complexity             vealed that many popular methods fail to estimate
inherent in lengthy textual inputs, making expla-           causal effects accurately and often perform no bet-
nations more intuitive and easier to communicate            ter than a naive concept-based matching baseline.
(Calderon and Reichart, 2024). Moreover, they nat-             Chaleshtori et al. (2024) recently noted an in-
urally support both local and global explanations.          creasing need for richer benchmarks that cap-
These advantages have driven their widespread use           ture the structural complexity of real-world data
in applications such as bias detection (Cornacchia          and enable the evaluation of both direct and in-
et al., 2023), providing clear and actionable expla-        direct causal effects. Complementing this, Du
nations (Bouchacourt and Denoyer, 2019), discov-            et al. (2025) demonstrated that even state-of-the-art
ering new hidden concepts (Ghorbani et al., 2019),          LLMs frequently fall prey to classical statistical

                                                        3
```

### Page 4

![Rendered page 4](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-004.png>)

#### Extracted Text

```text
      Box 2.1: Definitions: Causal Concept Effects and Estimators
      Definition 1 (Causal Concept Effect (CaCE) and Individual CaCE (ICaCE)).
                                           →
                                 CaCEf ( c ) = E f (X) | do(C = c′ ) − E [f (X) | do(C = c)]
                                                                     
                                           →
                             ICaCEf (xε , c ) = E f (X) | do(C = c′ , E = ε) − f (xε )
                                                                           

      Definition 2 (Empirical CaCE and ICaCE).
                                     \f →
                                                  1 X h  c∗ →c′      ∗ i
                                     CaCE      c =          f x̃ε∗   − f x̃cε∗ →c
                                                   |D| x ∈D
                                                        ε∗
                                        →        →
                                                    c
                                \ f xε , c =f x̃ε − f (xε )
                               ICaCE


fallacies, underscoring the limitations of existing              keeping all the exogenous variables fixed. In our
evaluation methods in assessing true causal reason-              setting, counterfactuals arise at two levels. First,
ing. We believe LIBERTy addresses these gaps                     a textual counterfactual is generated by propagat-
by simulating realistic scenarios with diverse text              ing the intervened concept assignment through the
types and rich causal graphs.                                    SCM. Second, a prediction counterfactual is ob-
                                                                 tained by passing this counterfactual text to the
3     Evaluation of Explanations                                 explained model and observing its new prediction.
In this section, we provide the relevant causal back-            Because the DGP is fully specified and LLM de-
ground and outline our causal approach to evaluat-               coding is deterministic (with the temperature set to
ing explanations of different scopes. Local expla-               zero), the counterfactuals align with Pearl’s defini-
nations capture how a concept influences a model’s               tion of structural counterfactuals (Pearl, 2013).
prediction for a specific instance, whereas global
explanations capture its influence across the en-                Causal Effect of Concepts We consider two lev-
tire data distribution. We evaluate explanations by              els of causal effects: the Causal Concept Effect
comparing them with causal effects: local expla-                 (CaCE) (Goyal et al., 2019), analogous to an Av-
nations against individual-level effects and global              erage Treatment Effect, and the individual CaCE
explanations against population-level effects.                   (ICaCE), analogous to an Individual Treatment Ef-
                                                                 fect, where the treatment is a concept, and the out-
3.1     Causality Background                                     come is the model prediction. Ideally, a faithful
Structural Causal Models We adopt the Struc-                     explanation method would estimate the CaCE as
tural Causal Model (SCM) framework of Pearl                      a global explanation and the ICaCE as a local
(2009). An SCM consists of exogenous and en-                     explanation (Gat et al., 2023). Formally, let C de-
dogenous variables, together with structural equa-               note the concept whose value changes from c to c′
                                                                          →
tions. Each endogenous variable is defined as a                  (written c ), and let E be exogenous variables with
function of its parent endogenous variables and its              ε values. Then, xε is the resulting text , and the pre-
associated exogenous noise variable. The induced                 diction of the explained model is f (xε ), which is a
causal graph is a directed acyclic graph encoding                vector with softmax probabilities of each class of
these dependencies. An example of a causal graph                 the concept Y the model f predicts. We denote ex-
is given in Figure 1. In this figure, the endoge-                pectations under the interventional distribution by
nous variables are the concepts (A, B, C, Y ), the               the standard do-operator notation E [·|do(C = c′ )]
LLM-generated text xε , and the prediction of the                (Pearl, 2009). The formal definitions are provided
explained model f (xε ). The exogenous variables                 in Box 2.1 Def 1. Both CaCE and ICaCE are vec-
include Gaussian noise terms (εa , εb , εc , εy ), or ran-       tors, capturing effects on all classes of Y .
domly sampled auxiliary text provided to the LLM
(εtemplate and εpersona ). Complementing the SCM                 3.2   Evaluating Explanations
with explicit distributions over the exogenous vari-             Estimating Causal Effects Both the CaCE and
ables yields the data-generating process (DGP).                  ICaCE are theoretical quantities, and in practice,
Counterfactuals Within the SCM framework, a                      we estimate them using counterfactuals. For xε ,
                                                                                                     →
counterfactual is the outcome of an intervention                 we denote its counterfactual by x̃εc . The for-
that assigns a different value to a concept while                mal definitions of the estimators are provided in

                                                             4
```

### Page 5

![Rendered page 5](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-005.png>)

#### Extracted Text

```text
   Box 3.1: Definitions: Evaluation Measures
   Definition 3 (ICaCE Error Distance (ED)).
                                          →
                                                      \ f (xε , →            →
                                                                               
                          ED f, Mf , xε , c = dist ICaCE        c ); Mf (xε , c )
                                                 1 X 1      X                      →
                                                                                      
                                  ED (f, Mf ) =                     ED f, Mf , xε , c
                                                |C| → |D→ |
                                                        c x ∈D
                                                              c ∈C           ε       →
                                                                                     c
   Definition 4 (ICaCE Order-Faithfulness (OF)).
                           → →
                                                                                                               
                                                                \ f (xε , c→2 ); Mf (xε , c→1 ) − Mf (xε , c→2 )
                                             \ f (xε , c→1 ) − ICaCE
            OF f, Mf , xε , c1 , c2 = sign ICaCE
                                            1       X           1         X                           → →
                                                                                                                
                      OF (f, Mf ) =                                                 OF f, Mf , xε , c1 , c2
                                      |C|(|C| − 1) → → |Dc→ ∩ Dc→ | x ∈D ∩D
                                                        c1 ,c2 ∈C        1       2       ε   →    →
                                                                                             c1   c2
                                                         c1 ̸=c2
   Where dist(·; ·) is a distance metric and sign(·; ·) is the proportion of vector entries that agree in sign.


Box 2.1 Def 2. In our setting, ICaCE\ f is exact                         tant than another if and only if its true causal effect
because, with fixed E and deterministic
                                    →  decoding,                       is larger. While ED measures estimation accuracy,
                     ′
E [f (X) | do(C=c , E=ε)] = f x̃εc . If decod-                           Order-Faithfulness assesses whether explanations
ing is stochastic (e.g., temperature > 0), additional                    preserve the relative ordering of concept impor-
noise is introduced through token sampling (see                          tance, a property that is often more robust, inter-
the discussion in Appendix A).                                           pretable, and directly relevant to how explanations
                                                                         are used in practice. To formalize this idea, con-
                                                                                                      →       →
Evaluation Pipeline The explained model f                                sider two concept changes c1 and c2 . We first com-
is trained on DGP-sampled data Df .3 The                                 pute the difference between their reference effect
explanation method M is trained on pairs                                 vectors, and then the difference between their expla-
(x, f (x)) : x ∈ DM , with optional access to gold                       nation vectors. We compare the signs of each entry
concept values or other auxiliary information, de-                       in the difference vector with the corresponding en-
pending on the evaluator choice. For evaluating                          try in the explanation difference vector. Agreement
Mf , we use the interventional test set DC , where                       of signs indicates that the explanation preserves the
C denotes the set of concept changes. For each                           correct ordering of the two concept changes, and
change, D→   c
               consists of pairs of textual examples                     is therefore order-faithful. The formal definition
                                          →
and their counterfactuals, (xεn, x̃εc ). From these,                     is provided in Box 3.1 Def 4. To summarize, we
               \ f (→             \ f (x, →                              report the average error distance ED (lower is bet-
                                                 o
we compute CaCE         c ) and ICaCE         c) ,
                                                  x                      ter) and the average order-faithfulness OF (higher
as well as the corresponding explanation     scores:                     is better) to compare explanation methods.
     →                                     →
                                 n           o
Mf ( c ) for global methods and Mf (x, c ) for
                                               x
local ones. We next describe the evaluation mea-                         4   Interventional Data Generation
sures for local explanations, which can be extended                      We next describe the process for generating an
to global explanations with minor modifications.                         interventional benchmark using LIBERTy (LLM-
Evaluation Measures CEBaB reports the aver-                              based Interventional Benchmark for Explainability
age ICaCE Error Distance over all the concept                            with Reference Targets). The framework relies on
changes, defined as the distance between the refer-                      explicitly defined DGPs comprising three compo-
ence effects and the explanation (formal definition                      nents: SCMs over concepts, exogenous grounding
in Box 3.1 Def 3). Following Abraham et al. (2022),                      texts, and an LLM (see Figure 1). These DGPs
we consider three distance metrics: cosine distance,                     allow us to generate silver counterfactuals.
L2 distance, and norm difference. We use their
                                                                         LIBERTy SCMs For each dataset, we first de-
mean as the final reported error distance (ED).
                                                                         fine the causal graph that specifies the concepts
   In addition, we propose a new measure, which
                                                                         and their directional relationships (which concept
we call Order-Faithfulness. This measure builds
                                                                         influences which). Based on this graph, we specify
on the necessary condition for faithful explanations
                                                                         the structural equations: each concept is linked to
introduced by Gat et al. (2023), which states that an
                                                                         a function that determines its value based on its
explanation must rank one concept as more impor-
                                                                         parent concepts and an exogenous noise term. The
   3
       No training is required if f is a zero/few-shot LLM.              noise is drawn from a Gaussian distribution, with a

                                                                     5
```

### Page 6

![Rendered page 6](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-006.png>)

#### Extracted Text

```text
   Wo r k pl ac e Vi o l e nc e Pr e di c t i o n                       Di s e as e De t e c t i o n                             CV Sc r e e ni ng

       Gender             Race                  Age          Dizzines            Facial Pain           Weakness
                                                                                                                        Gender           Race           Age

     Department           License          Tenure            Migrane              Sinusitis            Iinfluenza       Socio-                         Work
                                                                                                                       economic
                                                                                                                                       Education     experience

               Violence             Seniority                            Headache              Fever
                                                                                                                      Volunteering      Quality      Certificates
                                                               Light                Nasal
                                                             sensitivity          congestion
                     Demogr aphics                                                                                                   Demogr aphics
     Violence                                    Text                                                                                                    Text
       Label
                           Car eer                   HR
                                                 interview
                                                             Disease              Symptom                Text         Quality         Backgr ound         Personal
                                                                                                                                                         statement
                           Concepts                            Labels               Concepts             Forum post    Label             Concepts




Figure 2: LIBERTy Causal Graphs: We show only the concepts (endogenous variables) and the relationships
between them. Colored concepts indicate the variables that the explained model is trained to predict (the Y ). At the
bottom, a simplified version is provided. The graphs are grounded in prior literature and studies.

concept-specific mean and variance. The structural                                      ond, this decoding yields highly generic, templated,
equation generating the text takes all concepts as                                      and repetitive texts, regardless of concept values
inputs and uses two exogenous grounding texts, a                                        (always the same narrative, albeit with minor varia-
persona and a template, instead of Gaussian noise.                                      tions). Third, the generated examples do not seem
   In Figure 2, we illustrate the three causal graphs                                   like authentic human-written text. To address these
of the three LIBERTy datasets. While their SCMs                                         limitations, we propose a simple yet elegant so-
are not intended to mirror the true causal structure                                    lution in the spirit of the SCM framework: we
of the world (see the discussion in Appendix A.1),                                      introduce two additional exogenous variables, an
they are grounded in plausible assumptions: one                                         author persona and a template, both of which serve
causal graph (workplace violence prediction) is                                         as a grounding context for the LLM.
adapted from prior literature (Gerberich et al.,                                           The Persona variable εpersona represents a set of
2004), and the other two (disease detection and                                         contextual attributes, including profession, hobbies,
CV screening) are informed by statistical patterns                                      and personal motivations. In contrast, the Template
in real-world data (Monto et al., 2000; Cady and                                        variable εtemplate captures a particular discourse
Schreiber, 2002; Dastin, 2018). Finally, we note                                        structure, derived from real-world corpora (e.g.,
that our three causal graphs are much more com-                                         personal statements, Reddit posts). Templates and
plex and richer than the (four-concepts) causal                                         personas support three key goals: (1) making the
graph of CEBaB. Each graph includes at least                                            generated texts resemble authentic text; (2) pro-
eight concepts, exhibits confounding and media-                                         moting diversity: for each set of concept values,
tion structures (allowing estimation of direct and                                      there are |Epersona | × |Etemplate | possible instantia-
indirect effects), contains long paths (up to four                                      tions; and (3) ensuring the original example and its
edges between a concept and the text), and sup-                                         counterfactual derive from the same narrative.
ports both anticausal (Y → T ext) and confounded
                                                                                        Text Generation We sample concept values in
(Y ← C → T ext) learning problems.
                                                                                        topological order from the SCM, using the equa-
                                                                                        tions and Gaussian noise. We then sample a per-
Exogenous Grounding Texts             To ensure the
                                                                                        sona and a template and record all variable values
validity of our structural counterfactuals, determin-
                                                                                        for later counterfactual generation. Textual real-
istic decoding is required. With stochastic decod-
                                                                                        izations are generated via deterministic decoding
ing, generation noise cannot be tracked or held
                                                                                        (zero temperature) by conditioning GPT-4o on the
fixed across factual and counterfactual texts, caus-
                                                                                        full set of concept values, along with the persona
ing them to differ in unobserved exogenous factors
                                                                                        and template. We use a dedicated prompt for each
rather than only in the intervened concepts, and
                                                                                        dataset. Notably, GPT-4o receives only the concept
thus violating the definition of a structural coun-
                                                                                        values and does not observe the causal graph itself.
terfactual (see Appendix A.2). However, this re-
quirement introduces its own limitations. First, for                                    Counterfactual Generation       We follow Pearl’s
a given combination of concept values, determin-                                        three-step counterfactual procedure (Pearl, 2009).
istic decoding produces a single, fixed text. Sec-                                      (1) Abduction: fix the exogenous variables used

                                                                                    6
```

### Page 7

![Rendered page 7](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-007.png>)

#### Extracted Text

```text
    Dataset                  D→     Pairs    Words              10 opening and 10 closing sentence variants are de-
                              c
                                                                fined to maintain a coherent interview flow. Each
    Workplace Violence      1756     1317     350.9
                                                                template is generated by sampling one question per
    Disease Detection       1243      932     310.8
                                                                concept, along with an opening and closing sen-
    CV Screening            1332      998     313.0
                                                                tence. The question order is randomized, yielding
Table 1: Data Statistics: For all datasets, |Df |=1.5K          a large pool of interview templates. The persona
                                                →               contains three informal “fun facts” about the nurse,
and |DM |=0.5K. Pairs is the number of (xε , x̃εc ) pairs
in D→  . Words reports the average number per example.          each centered on a concept (without specifying its
     c
                                                                value). Using Gemini, we generated 500 personas.
for the original example, (2) Action: intervene on a            Additional details are in Appendix D.1.
target concept, and (3) Prediction: propagate the in-
                                                                5.2    Disease detection
tervention through the SCM and compute updated
concept values. We then regenerate the text using               This dataset simulates clinical self-reports, where
the same persona, template, and deterministic de-               the (explained) model predicts a disease from symp-
coding. The red arrows in Figure 1 illustrate this.             toms described in a medical forum post. Unlike
For each test example, we randomly select three                 the other two datasets, the learning problem is anti-
concept changes and generate counterfactuals.                   causal: the disease label serves as the root cause
                                                                in the SCM and determines the values of symp-
5     Datasets                                                  tom concepts, based on known symptom–disease
                                                                relations (Monto et al., 2000; Cady and Schreiber,
LIBERTy comprises three datasets, each modeling
                                                                2002). The template is a narrative structure ab-
a high-stakes, socially impactful NLP task where
                                                                stracted from 1,310 posts on Reddit’s DiagnoseMe
explainability is critical. Each dataset is divided
                                                                forum,4 using Gemini to preserve the clinical tone
into four subsets: two for training and testing the
                                                                and flow. The persona (a total of 1200) consists of
explained model, one for training the explanation
                                                                three informal facts about occupation, hobbies, and
method, and one test set containing pairs of texts
                                                                family or friends. To generate personas, we first
and their counterfactuals. The first three subsets
                                                                sample an occupation and a hobby from predefined
exclude counterfactuals, which are unavailable in
                                                                lists, then use Gemini to generate the corresponding
real-world settings. The number of examples in
                                                                facts. Each dataset example is created by prompt-
each dataset is provided in Table 1. The LLM
                                                                ing GPT-4o to follow the template and integrate
integrated within the SCMs (for generating texts)
                                                                information from the persona and the symptom
is GPT-4o, while Gemini-1.5-Pro is used to create
                                                                values. Additional details are in Appendix D.2.
templates and personas. Below, we briefly describe
each dataset. Due to space limitations, the SCMs,
                                                                5.3    CV Screening
prompts, representative examples, and additional
technical details are provided in Appendix D.                   This dataset simulates automated resume assess-
                                                                ment, where the model is tasked with predicting
5.1    Workplace Violence Prediction                            an applicant’s quality from a CV-style personal
This dataset simulates HR–nurse interviews, in                  statement, with labels such as weak, qualified, and
which the (explained) model predicts the likelihood             outstanding. Motivated by critiques of real-world
that a nurse will experience workplace violence.                screening systems (Dastin, 2018; Raghavan et al.,
The causal graph is adapted from the Minnesota                  2020; Cowgill et al., 2020), the causal graph en-
Nurses’ Study (Gerberich et al., 2004), which doc-              codes hypothesized dependencies between demo-
umented the prevalence of verbal and physical vio-              graphic and professional attributes, inspired by sta-
lence among clinical staff and analyzed risk factors            tistical patterns reported by the U.S. Bureau of
by demographic and professional background. The                 Labor Statistics.5 For example, gender influences
template follows a structured HR interview format.              the hiring label only indirectly through mediators
To ensure both realism and sufficient diversity, we             such as education and Work Experience. 1,235
generate interview templates as follows: for each               templates were generated from 342 scraped per-
concept, a bank of 10 questions is created using                   4
                                                                   https://www.reddit.com/r/DiagnoseMe/
Gemini, each designed to elicit the concept’s value                5
                                                                   https://www.bls.gov/cps/demographics.
from different linguistic perspectives. Additionally,           htm


                                                            7
```

### Page 8

![Rendered page 8](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-008.png>)

#### Extracted Text

```text
sonal statement examples,6 where each source text           which fixes confounders while allowing mediators
was abstracted with Gemini using a 2-shot prompt            to vary, and achieves the best performance.
to produce several occupation-agnostic variants                (2) Matching (see Appendix C.2): matching
that preserve the narrative structure while remov-          methods search for the most similar candidate from
ing concept- and role-specific details. To gener-           a predefined set of examples with the target concept
ate a persona (a total of 990), we sample a role            change. The difference between the methods lies
from a predefined list and use Gemini with a 2-shot         in how similarity is defined, and we examine five
prompt to produce both personal and professional            methods: (2a) ST Match: cosine similarity over
context, including motivations and skills relevant          SentenceTransformer embeddings (Reimers and
to that role. Each dataset example is then created          Gurevych, 2019); (2b) PT Match: cosine similarity
by prompting GPT-4o to follow the template and              over a pre-trained encoder-only model (DeBERTa);
integrate information from the application role, the        (2c) FT Match: cosine similarity over an encoder
persona, and the sampled concept values. Addi-              fine-tuned to predict Y ; (2d) Approx: first predicts
tional details are in Appendix D.3.                         concept values using fine-tuned models and then
                                                            search for exact concept-based match; and (2e)
6       Experimental Setup                                  ConVecs: cosine similarity over concatenated soft-
Using LIBERTy, we conduct experiments on five               max prediction vectors of all concepts. Notably,
explained models and benchmark eight explanation            the first three are semantic-based methods, while
methods from four families of approaches. The               the latter two are concept-based ones.
goals of our experiments are: (1) Benchmarking lo-             (3) Concept Erasure (see Appendix C.3): re-
cal and global explanation methods; (2) Analyzing           moves linearly encoded information about a target
the sensitivity of models to concept changes and            concept from hidden representations using LEACE
evaluating which model captures better the causal           (Belrose et al., 2023).7 (4) Concept Attributions
structure of the data. The evaluation pipeline is de-       (see Appendix C.4): estimates concept importance
scribed in Section 3.2. When reporting scores, we           via ConceptShap (Yeh et al., 2020) combined with
typically average them over all concept changes.            TCAV (Kim et al., 2018), which construct concept
                                                            vectors and assign Shapley-based scores.8
Explained Models We evaluate five models.
Three are fine-tuned to predict Y from text: (1)            7     Results
DeBERTa-v3 (base, He et al. (2020)), an encoder-
only model; (2) T5 (base, Raffel et al. (2020)), an         7.1    Local Explanations
encoder–decoder model; and (3) Qwen-2.5 (1.5B-              We begin by comparing the local explainability
instruct, Team (2023)), a decoder-only LLM. The             methods using LIBERTy, reporting ICaCE ED and
other two are zero-shot LLMs: (4) Llama-3.1 (8B-            OF. Table 2 presents these measures at the dataset
instruct, Dubey et al. (2024)) and (5) GPT-4o (Ope-         level (averaged across all five models) and at the
nAI, 2024). See Appendix E.2 for more details,              model level (averaged across all three datasets).
hyperparameters, performance, and prompts.                  Complete results are provided in Table 15 (Ap-
Explainability Methods We briefly mention                   pendix F). Overall, the matching approach per-
the explainability methods we benchmark, but Ap-            forms best. Within this category, FT Match, which
pendix C thoroughly describes and discusses them.           fine-tunes an encoder-only model to predict the la-
The rationale for selecting methods was to focus            bel Y and then uses its embeddings for similarity,
on top-performing approaches previously applied             achieves the lowest estimation error and emerges
to CEBaB with user-friendly code. We examine                as the most faithful method. Its advantage likely
eight methods covering four families:                       stems from the model learning task-specific rep-
   (1) Counterfactual Generation: LLMs generate             resentations that produce more meaningful neigh-
counterfactuals by editing texts to reflect a target        borhoods for matching. Other strong performers
concept change (Gat et al., 2023). We examine               are the concept-based matching methods, ConVecs
in Appendix C.1 four prompting techniques, each                 7
                                                                  We employ LEACE only for open-source models and only
injecting different causal assumptions. We mainly           on the Disease Detection dataset, where erasing a concept is
focus on the Mediators and Confounders technique,           well defined as its absence (e.g., symptom not present).
                                                                8
                                                                  We benchmark ConceptShap only as a global explanation
    6
        https://universitycompare.com                       for open-source models.


                                                        8
```

### Page 9

![Rendered page 9](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-009.png>)

#### Extracted Text

```text
                                                              Dataset                                  Explained Model
                  Average
                                          Violence            Disease              CV         DeBERTa-v3   Qwen-2.5     GPT-4o
    ↓ Method     ED           OF         ED     OF           ED    OF      ED           OF    ED    OF    ED     OF    ED    OF
    CF Gen      0.55         0.49       0.47          0.58   0.67   0.36   0.52      0.52     0.50     0.59       0.62     0.53    0.58      0.49
    Approx      0.45         0.69       0.41          0.71   0.48   0.69   0.46      0.66     0.38     0.76       0.50     0.70    0.53      0.67
    ConVecs     0.44         0.69       0.40          0.73   0.44   0.70   0.47      0.66     0.34     0.78       0.47     0.71    0.52      0.68
    ST Match    0.49         0.65       0.51          0.63   0.46   0.69   0.50      0.62     0.49     0.69       0.55     0.66    0.53      0.67
    PT Match    0.51         0.64       0.51          0.64   0.52   0.65   0.50      0.63     0.52     0.68       0.56     0.65    0.59      0.64
    FT Match    0.34         0.74       0.32          0.76   0.36   0.75   0.35      0.72     0.16     0.88       0.39     0.75    0.48      0.70
    LEACE       0.65         0.46           —         —      0.65   0.46       —        —     0.62     0.42       0.87     0.41        —     —

Table 2: Local Explainability Results: We report the Average ICaCE Error-Distance (ED ≥ 0, ↓ is better) and
Average ICaCE Order-Faithfulness (OF ≤ 1, ↑ is better). The Average column reports the mean across five
explained models and three datasets. The detailed results appear in Appendix Table 15 and exhibit a similar pattern,
with fine-tuned matching outperforming other approaches. Horizontal lines separate method families.

                                                                                   Dataset           Violence          Disease               CV
                                             0.85 FT Match                         Model             Qwen-2.5        DeBERTa-v3             GPT-4o
                                           0.82 ConVecs                                               Gender             Light Sens        Work Exp
                                         0.77 Approx                               Gold              Department          Facial Pain       Education
                                        0.73      ST Match                                              Age               Dizziness          Race
                                      0.69        PT Match                                            Gender             Light Sens        Education
                                   0.62           LEACE                            FT Match          Seniority            Dizziness        Work Exp
                                   0.61           CF Gen                                               Age               Facial Pain         Age
                      0.41                        ConceptSHAP                                         Gender             Weakness          Education
                                                                                   CF Gen              Age               Dizziness         Work Exp
    0.0   0.2   0.4          0.6      0.8       1.0                                                   Race               Light Sens        Socioeco
      Mean Global Order-Faithfulness                                                                                     Dizziness
Figure 3: Global Explainability Results: We report the                             LEACE                                 Light Sens
                                                                                                                         Headache
mean Order-Faithfulness score for global explanations.
See Table 16 in the Appendix for full results.                                                        Gender          Dizziness
                                                                                   ConceptShap         Race          Nasal Cong
                                                                                                     Seniority       Weakness
(proposed in this work) and Approx. These find-
ings align with those of Gat et al. (2023), who                                Table 3: Global Explanations Analysis: We present
compared different matching methods on CEBaB                                   the top-3 most important concepts of explanations for
and reported similar trends.                                                   selected datasets, models, and methods. A colored con-
                                                                               cept indicates it is among the top three gold concepts.
   An interesting difference between our findings
and those of Gat et al. (2023) is that, while LLM-
generated counterfactuals outperform matching-                                 room for improvement. In LIBERTy, even the best
based methods on CEBaB, the opposite holds on                                  methods achieve only around 0.3 on ED (where 0
LIBERTy. A potential explanation is that humans                                is perfect) and 0.7 on OF (where 1 is perfect). We
write CEBaB’s counterfactuals: annotators edit an                              hope that LIBERTy will encourage further progress
existing text to reflect a change in concept. LLMs                             on developing more faithful explanation methods.
can closely mimic this editing process, especially
for short, simple texts, which makes their gener-                              7.2        Global Explanations
ated counterfactuals appear effective. In LIBERTy,                             Many global explanations produce a ranked list
producing an explanation that resembles a human                                of concepts by their overall importance (not spe-
edit does not guarantee faithfulness; instead, the                             cific to a single example), reflecting their influence
explanation should reflect the actual DGP. This                                on the model’s predictions (a.k.a. feature impor-
also explains why matching methods perform more                                tance). We therefore evaluate their global order-
consistently: their retrieved candidates are sam-                              faithfulness: are the concepts ranked in the same
pled from distributions aligned with the underlying                            order as their causal effects? To obtain the ground-
DGP rather than produced through human-aligned                                 truth ranking, we compute a single gold impor-
textual edits. We refer the reader to an extended                              tance score for each concept using CaCE. Note that
discussion of these aspects in Appendix A.3.                                   for each concept change, CaCE yields a vector of
   Finally, the ED and OF scores reveal substantial                            size |Y |, capturing the causal effect of that change

                                                                           9
```

### Page 10

![Rendered page 10](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-010.png>)

#### Extracted Text

```text
   Examined            Workplace Violence                   Disease Detection                          CV Screening
   Model              Race  Gender     Age             Headache   General Weakness              Race      Gender    Age
   DeBERTa-v3        0.350       1.192      0.758        0.398                  0.415          0.715      0.432     0.613
   T5                0.421       0.743      0.512        0.530                  0.376          0.742      0.398     0.513
   Qwen-2.5          0.691       1.314      1.045        0.426                  0.512          0.522      0.361     0.503
   Llama-3.1         0.224       0.227      0.226        0.364                  0.332          0.374      0.283     0.397
   GPT-4o            0.724       0.594      0.300        0.369                  0.215          0.417      0.208     0.355
   True Effect       0.484       1.271      1.154          –                      –            0.636      0.369     0.913

Table 4: Concept Sensitivity Analysis: In the Disease Detection dataset, Y is the parent of the concepts, so
interventions do not affect Y , and its ground-truth sensitivity cannot be computed. See Table 17 for full results.

on each output class. To obtain a single gold im-                     reflect the underlying causal relationships. Under
portance score for each concept, we first sum the                     successful learning, a model’s sensitivity to con-
absolute CaCE values across all output classes, re-                   cept changes should closely match the true causal
flecting the total magnitude of the effect for that                   effects on the outcome variable Y , which we esti-
change. We then average this quantity over all                        mate via Monte Carlo simulation from the SCM.
changes. This produces a single gold importance                          For a given example and concept change, we
score per concept. Global OF is then computed                         compute a sensitivity score that quantifies the ex-
based on these scores: it quantifies how faithfully                   tent to which the model’s prediction is affected.
each explanation method’s ranking of concept im-                      This score is obtained by summing the absolute
portance matches the gold ranking.9                                   ICaCE values, which quantify the magnitude of the
   Figure 3 compares the methods using the aver-                      change. Larger values indicate stronger shifts in the
age global OF across the three datasets and five                      prediction (i.e., more sensitive). Table 4 reports sen-
models. The complete (non-averaged) results are                       sitivity scores for the five evaluated models on se-
reported in Table 16 in the Appendix. As shown,                       lected concepts (an average over all their changes),
global trends mirror the local ones, with the match-                  alongside the gold sensitivity effect. Complete re-
ing approach outperforming the others. Table 3                        sults are provided in Table 17 in the Appendix.
further reports the top-3 most important concepts                        When examining sensitivity scores (without
identified by each method and compares them with                      comparing them to the gold effects) we observe
the top-3 gold concepts (according to their gold                      that zero-shot LLMs (Llama-3.1-8B and GPT-4o)
importance score). Every method misses at least                       exhibit lower sensitivity to concept changes, par-
one gold concept, highlighting the need for further                   ticularly for demographic concepts such as Race,
research on global explainability.                                    Gender, and Age (Table 4). We believe the reduced
                                                                      sensitivity reflects intentional design choices made
7.3    Sensitivity Analysis                                           during post-training alignment. In addition, among
Up to this point, we have used LIBERTy to evaluate                    the fine-tuned models, we find that Qwen2.5-1.5B
explanation methods. More broadly, the framework                      most accurately reflects the causal structure of the
supports two complementary analyses. First, it can                    data. Still, the gap with the gold effects highlights
be used to analyze a model’s sensitivity to concept                   that fine-tuning is insufficient and that there re-
changes by measuring the magnitude of predic-                         mains a need for causal learning techniques.
tion changes induced by structural counterfactuals.
Second, LIBERTy can be used to assess how well                        8   Conclusions
different learning methods, such as CE fine-tuning,                   A central challenge in explainability is the lack of
align model behavior with the causal structure en-                    reliable evaluation protocols, particularly given the
coded in the DGP. This second analysis necessarily                    absence of “gold explanations”. Our work takes
focuses on models trained on the generated data,                      a significant step toward closing this gap. We in-
since only then can their behavior be expected to                     troduced LIBERTy, a framework for generating in-
   9
     Global OF and ICaCE OF differ both in the order of               terventional datasets to benchmark concept-based
computation and in what is being ranked. ICaCE OF evalu-              explanations against “silver” references: causal
ates order-faithfulness over individual concept changes on a          effects estimated using structural counterfactuals.
per-example basis before averaging, whereas Global OF evalu-
ates order-faithfulness over concepts, using global importance        Using LIBERTy, we evaluated local and global ex-
scores derived from CaCE.                                             plainability methods, the sensitivity of LLMs to

                                                                 10
```

### Page 11

![Rendered page 11](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-011.png>)

#### Extracted Text

```text
concept interventions, and the causal learning ca-             free-text explanations (see the analysis of Calderon
pabilities of fine-tuned models.                               and Reichart (2024)). Nevertheless, there are
   In Section A.4 in the Appendix, we outline fu-              strong reasons to focus on concept-based methods.
ture research opportunities motivated by our four              These methods quantify how high-level, human-
key findings. First, we found that LLM-generated               interpretable concepts (a.k.a. attributes, features,
counterfactuals, which were previously reported as             variables, or rubrics) that are implicitly or explic-
state-of-the-art explanations (Gat et al., 2023), do           itly expressed in text influence the model. Because
not retain this status when evaluated against struc-           such high-level concepts align with human cogni-
tural counterfactuals (as in LIBERTy) rather than              tive processes (Alqaraawi et al., 2020; Kim et al.,
human-written ones (as in CEBaB). This highlights              2022; Poeta et al., 2023), reduce the complexity
the need for a broader evaluation of explanations.             of long inputs, and communicate model behavior
Second, we observed a large room for improvement               in intuitive terms (Calderon and Reichart, 2024),
in both local and global explanations, offering clear          concept-based explanations are particularly suit-
targets for future work.                                       able for high-stakes settings where end users and
   Third, our concept-sensitivity analysis showed              decision makers must understand and trust model
that some LLMs are largely insensitive to demo-                reasoning. We believe that the relatively limited at-
graphic interventions, likely due to post-alignment            tention to concept-based explainability stems partly
mitigation effects. Finally, our analysis revealed             from the lack of appropriate benchmarks for devel-
that vanilla fine-tuning may fail to capture the               oping and evaluating such methods. By providing
causal structure of the data, suggesting the need              an interventional benchmark with structural causal
for unique learning methods. To summarize, there               effects, LIBERTy aims to address this gap and fa-
is great promise in developing smaller, theory-                cilitate broader research and adoption of concept-
grounded, causal-inspired explainability and learn-            based explanations.
ing approaches. We hope our work will serve as a
foundation for such future research.                           DGPs as Approximations of Reality LIBERTy
                                                               provides structural counterfactuals in the strict
9   Limitations                                                sense, as they are generated from a fully specified
                                                               data-generating process (DGP). While the DGP
Synthetic Text Generation LIBERTy relies on
                                                               and causal graphs only simplify real-world mech-
LLMs to instantiate structural counterfactuals.
                                                               anisms, they are not arbitrary and are grounded
However, it also means that the texts are synthetic
                                                               in domain knowledge and the literature. Still,
rather than human-written. This may introduce mis-
                                                               we acknowledge that they do not perfectly mir-
matches between how the LLM instantiates con-
                                                               ror real-world causal structures. Crucially, this
cepts and how humans would naturally express
                                                               limitation does not compromise the reliability of
them. To assess data quality, we conducted a
                                                               our evaluation protocol, because our goal is not
human evaluation (Appendix B). Annotators con-
                                                               to recover real-world mechanisms or estimate real-
firmed that the generated texts are coherent, rele-
                                                               world causal effects. Instead, our objective is to
vant, and fluent; that the LLM correctly incorpo-
                                                               measure the causal effects within the explained
rates concept values; and that counterfactuals are
                                                               model and benchmark explanation methods against
perceived as realistic variants differing in only one
                                                               those effects. For this purpose, what matters is
concept. Finally, although LIBERTy uses synthetic
                                                               that the DGP supports precise interventions and
text, this limitation is increasingly less restrictive:
                                                               produces structural counterfactuals that faithfully
a growing share of real-world data is generated
                                                               reflect them. Explanation faithfulness is always
by LLMs, making synthetic settings both common
                                                               defined relative to the explained model, whether its
and practically meaningful. It is therefore reason-
                                                               behavior arises from true causal relationships, sim-
able to assume that model inputs in many future
                                                               plified abstractions, or even spurious correlations.
applications will themselves be LLM-generated.
                                                               Thus, a synthetic DGP is sufficient and, in practice,
Focusing on Concept-based Explanations Our                     often required for the controlled and rigorous evalu-
work focuses exclusively on concept-based expla-               ation of explanation methods. Such methods can be
nations and their causal evaluation. This scope                trained on or applied to data generated by the DGP
covers only a subset of existing explainability meth-          and evaluated against the explained model’s predic-
ods, and most prior work centers on token-level or             tions, whether or not the model itself was trained on

                                                          11
```

### Page 12

![Rendered page 12](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-012.png>)

#### Extracted Text

```text
that data. We do not claim that our benchmark ex-              Ivi Chatzi, Nina L. Corvelo Benz, Eleni Straitouri,
plains real-world phenomena or reveals how LLMs                   Stratis Tsirtsis, and Manuel Gomez-Rodriguez. 2025.
                                                                  Counterfactual token generation in large language
internally represent them. Rather, our goal is to
                                                                  models. In Causal Learning and Reasoning, Lau-
provide a principled benchmark for comparing ex-                  sanne, Switzerland, 7-9 May 2025, volume 275 of
planation methods, analyzing their limitations, and               Proceedings of Machine Learning Research, pages
identifying those that most faithfully capture model              1291–1315. PMLR.
behavior, thereby enabling their application in real-          Giandomenico Cornacchia, Vito Walter Anelli, Fedelu-
world settings. Please also see our discussion in                cio Narducci, Azzurra Ragone, and Eugenio Di Sci-
Appendix A.1.                                                    ascio. 2023. Counterfactual reasoning for bias evalu-
                                                                 ation and detection in a fairness under unawareness
Acknowledgments                                                  setting. CoRR, abs/2302.08204.
                                                               Bo Cowgill, Fabrizio Dell’Acqua, Samuel Deng, Daniel
References                                                       Hsu, Nakul Verma, and Augustin Chaintreau. 2020.
                                                                 Biased programmers? or biased data? A field
Eldar David Abraham, Karel D’Oosterlinck, Amir                   experiment in operationalizing AI ethics. CoRR,
  Feder, Yair Ori Gat, Atticus Geiger, Christopher               abs/2012.02394.
  Potts, Roi Reichart, and Zhengxuan Wu. 2022.
  Cebab: Estimating the causal effects of real-                Fahim Dalvi, Abdul Rafae Khan, Firoj Alam, Nadir
  world concepts on NLP model behavior. CoRR,                    Durrani, Jia Xu, and Hassan Sajjad. 2022. Dis-
  abs/2205.14140.                                                covering latent concepts learned in BERT. CoRR,
                                                                 abs/2205.07237.
Ahmed Alqaraawi, Martin Schuessler, Philipp Weiß, En-
  rico Costanza, and Nadia Berthouze. 2020. Evaluat-           Jeffrey Dastin. 2018. Amazon scrapped ‘ai’ recruiting
  ing saliency map explanations for convolutional neu-            tool that showed bias against women.
  ral networks: A user study. CoRR, abs/2002.00772.
                                                               Jin Du, Li Chen, Xun Xian, An Luo, Fangqiao Tian,
Esma Balkir, Svetlana Kiritchenko, Isar Nejadgholi, and           Ganghua Wang, Charles Doss, Xiaotong Shen, and
  Kathleen C. Fraser. 2022. Challenges in applying                Jie Ding. 2025. Ice cream doesn’t cause drown-
  explainability methods to improve the fairness of               ing: Benchmarking llms against statistical pitfalls
  NLP models. CoRR, abs/2206.03945.                               in causal inference. CoRR, abs/2505.13770.
                                                               Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Nora Belrose, David Schneider-Joseph, Shauli Ravfo-
                                                                 Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
  gel, Ryan Cotterell, Edward Raff, and Stella Bider-
                                                                 Akhil Mathur, Alan Schelten, Amy Yang, Angela
  man. 2023. LEACE: perfect linear concept erasure
                                                                 Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang,
  in closed form. CoRR, abs/2306.03819.
                                                                 Archi Mitra, Archie Sravankumar, Artem Korenev,
Kenza Benkirane, Jackie Kay, and María Pérez-Ortiz.              Arthur Hinsvark, Arun Rao, Aston Zhang, and 82
  2024. How can we diagnose and treat bias in                    others. 2024. The llama 3 herd of models. CoRR,
  large language models for clinical decision-making?            abs/2407.21783.
  CoRR, abs/2410.16574.                                        Amir Feder, Nadav Oved, Uri Shalit, and Roi Reichart.
                                                                2021. Causalm: Causal model explanation through
Diane Bouchacourt and Ludovic Denoyer. 2019.
                                                                counterfactual language models. Comput. Linguis-
  EDUCE: explaining model decisions through
                                                                tics, 47(2):333–386.
  unsupervised concepts extraction.   CoRR,
  abs/1905.11852.                                              Yair Ori Gat, Nitay Calderon, Amir Feder, Alexan-
                                                                 der Chapanin, Amit Sharma, and Roi Reichart.
Roger K Cady and Curtis P Schreiber. 2002. Si-                   2023. Faithful explanations of black-box NLP mod-
  nus headache or migraine? considerations in                    els using llm-generated counterfactuals. CoRR,
  making a differential diagnosis.   Neurology,                  abs/2310.00603.
  58(9_suppl_6):S10–S14.
                                                               S Gerberich, T Church, P McGovern, and et al. 2004.
Nitay Calderon, Liat Ein-Dor, and Roi Reichart. 2025.            An epidemiological study of the magnitude and con-
  Multi-domain explainability of preferences. CoRR,              sequences of work related violence: the minnesota
  abs/2505.20088.                                                nurses’ study. Occupational and Environmental
                                                                 Medicine, 61(6):495–503.
Nitay Calderon and Roi Reichart. 2024. On behalf of
  the stakeholders: Trends in NLP model interpretabil-         Amirata Ghorbani, James Wexler, James Y. Zou, and
  ity in the era of llms. CoRR, abs/2407.19200.                 Been Kim. 2019. Towards automatic concept-based
                                                                explanations. In Advances in Neural Information
Fateme Hashemi Chaleshtori, Atreya Ghosal, Alexander            Processing Systems 32: Annual Conference on Neu-
  Gill, Purbid Bambroo, and Ana Marasovic. 2024. On             ral Information Processing Systems 2019, NeurIPS
  evaluating explanation utility for human-ai decision          2019, December 8-14, 2019, Vancouver, BC, Canada,
  making in NLP. CoRR, abs/2407.03545.                          pages 9273–9282.


                                                          12
```

### Page 13

![Rendered page 13](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-013.png>)

#### Extracted Text

```text
Yash Goyal, Uri Shalit, and Been Kim. 2019. Ex-                  Sören R. Künzel, Jasjeet S. Sekhon, Peter J. Bickel, and
  plaining classifiers with causal concept effect (cace).          Bin Yu. 2019. Meta-learners for estimating hetero-
  CoRR, abs/1907.07165.                                            geneous treatment effects using machine learning.
                                                                   arXiv.
Riccardo Guidotti, Anna Monreale, Franco Turini, Dino
  Pedreschi, and Fosca Giannotti. 2018. A survey of              Jun Rui Lee, Sadegh Emami, Michael David Hollins,
  methods for explaining black box models. CoRR,                   Timothy C. H. Wong, Carlos Ignacio Villalobos
  abs/1802.01933.                                                  Sánchez, Francesca Toni, Dekai Zhang, and Adam
                                                                   Dejl. 2025. Xai-units: Benchmarking explainability
Sai Gurrapu, Ajay Kulkarni, Lifu Huang, Ismini                     methods with unit tests. CoRR, abs/2506.01059.
  Lourentzou, and Feras A. Batarseh. 2023. Ratio-
  nalization for explainable NLP: a survey. Frontiers            Yongqi Li, Mayi Xu, Xin Miao, Shen Zhou, and Tieyun
  Artif. Intell., 6.                                               Qian. 2024. Prompting large language models for
                                                                   counterfactual generation: An empirical study. In
Peter Hase and Mohit Bansal. 2020. Evaluating explain-             Proceedings of the 2024 Joint International Confer-
  able AI: which algorithmic explanations help users               ence on Computational Linguistics, Language Re-
  predict model behavior? CoRR, abs/2005.01831.                    sources and Evaluation, LREC/COLING 2024, 20-25
Pengcheng He, Xiaodong Liu, Jianfeng Gao, and                      May, 2024, Torino, Italy, pages 13201–13221. ELRA
  Weizhu Chen. 2020.        Deberta: Decoding-                     and ICCL.
  enhanced BERT with disentangled attention. CoRR,               Scott M. Lundberg and Su-In Lee. 2017. A unified
  abs/2006.03654.                                                  approach to interpreting model predictions. CoRR,
Anna Hedström, Philine Lou Bommer, Kristoffer Knut-                abs/1705.07874.
  sen Wickstrøm, Wojciech Samek, Sebastian La-                   Siwen Luo, Hamish Ivison, Soyeon Caren Han, and
  puschkin, and Marina M.-C. Höhne. 2023. The meta-                Josiah Poon. 2024. Local interpretations for explain-
  evaluation problem in explainable AI: identifying                able natural language processing: A survey. ACM
  reliable estimators with metaquantus. Trans. Mach.               Comput. Surv., 56(9):232:1–232:36.
  Learn. Res., 2023.
                                                                 Qing Lyu, Marianna Apidianaki, and Chris Callison-
XinYue Jiang, Jingsong He, and Li Gu. 2025. MTCR:                  Burch. 2022. Towards faithful model explanation in
  method for matching texts against causal relationship.           NLP: A survey. CoRR, abs/2209.11326.
  Neural Process. Lett., 57(3):58.
                                                                 Arnold S Monto, Stefan Gravenstein, Michael Elliott,
Enkelejda Kasneci, Kathrin Sessler, Stefan Küche-                  Michael Colopy, and Jo Schweinle. 2000. Clinical
  mann, Maria Bannert, Daryna Dementieva, Frank                    signs and symptoms predicting influenza infection.
  Fischer, Urs Gasser, Georg Groh, Stephan Günne-                  Archives of internal medicine, 160(21):3243–3247.
  mann, Eyke Hüllermeier, Stephan Krusche, Gitta
  Kutyniok, Tilman Michaeli, Claudia Nerdel, Jürgen              Yuqi Nie, Yaxuan Kong, Xiaowen Dong, John M. Mul-
  Pfeffer, Oleksandra Poquet, Michael Sailer, Albrecht             vey, H. Vincent Poor, Qingsong Wen, and Stefan
  Schmidt, Tina Seidel, and 2 others. 2023. Chatgpt                Zohren. 2024. A survey of large language models
  for good? on opportunities and challenges of large               for financial applications: Progress, prospects and
  language models for education. ScienceDirect.                    challenges. CoRR, abs/2406.11903.
Been Kim, Martin Wattenberg, Justin Gilmer, Carrie J.            OpenAI. 2024. Gpt-4o technical report. arXiv preprint.
  Cai, James Wexler, Fernanda B. Viégas, and Rory
  Sayres. 2018. Interpretability beyond feature attri-           Judea Pearl. 2009. Causality: Models, Reasoning, and
  bution: Quantitative testing with concept activation             Inference, 2 edition. Cambridge University Press.
  vectors (TCAV). In Proceedings of the 35th Inter-              Judea Pearl. 2013. Structural counterfactuals: A brief
  national Conference on Machine Learning, ICML                    introduction. Cognitive science, 37(6):977–985.
  2018, Stockholmsmässan, Stockholm, Sweden, July
  10-15, 2018, volume 80 of Proceedings of Machine               Lotem Peled-Cohen, Maya Zadok, Nitay Calderon,
  Learning Research, pages 2673–2682. PMLR.                        Hila Gonen, and Roi Reichart. 2025. Dementia
                                                                   through different eyes: Explainable modeling of
Sunnie S. Y. Kim, Elizabeth Anne Watkins, Olga                     human and LLM perceptions for early awareness.
  Russakovsky, Ruth Fong, and Andrés Monroy-                       CoRR, abs/2505.13418.
  Hernández. 2022. "help me help the ai": Under-
  standing how explainability can support human-ai               Eleonora Poeta, Gabriele Ciravegna, Eliana Pastor,
  interaction. CoRR, abs/2210.03735.                               Tania Cerquitelli, and Elena Baralis. 2023. Concept-
                                                                   based explainable artificial intelligence: A survey.
Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen                 CoRR, abs/2312.12936.
  Mussmann, Emma Pierson, Been Kim, and Percy
  Liang. 2020. Concept bottleneck models. In Pro-                Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
  ceedings of the 37th International Conference on                 Lee, Sharan Narang, Michael Matena, Yanqi Zhou,
  Machine Learning, ICML 2020, 13-18 July 2020, Vir-               Wei Li, and Peter J. Liu. 2020. Exploring the limits
  tual Event, volume 119 of Proceedings of Machine                 of transfer learning with a unified text-to-text trans-
  Learning Research, pages 5338–5348. PMLR.                        former. J. Mach. Learn. Res., 21:140:1–140:67.


                                                            13
```

### Page 14

![Rendered page 14](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-014.png>)

#### Extracted Text

```text
Manish Raghavan, Solon Barocas, Jon M. Kleinberg,               Victor Veitch, Dhanya Sridhar, and David M. Blei. 2020.
 and Karen Levy. 2020. Mitigating bias in algorithmic             Adapting text embeddings for causal inference. In
 hiring: evaluating claims and practices. In FAT* ’20:            Proceedings of the Thirty-Sixth Conference on Un-
 Conference on Fairness, Accountability, and Trans-               certainty in Artificial Intelligence, UAI 2020, virtual
 parency, Barcelona, Spain, January 27-30, 2020,                  online, August 3-6, 2020, volume 124 of Proceed-
 pages 469–481. ACM.                                              ings of Machine Learning Research, pages 919–928.
                                                                  AUAI Press.
Shauli Ravfogel, Anej Svete, Vésteinn Snæbjarnarson,
  and Ryan Cotterell. 2025. Gumbel counterfactual               Lijie Wang, Yaozong Shen, Shuyuan Peng, Shuai Zhang,
  generation from language models. In The Thirteenth               Xinyan Xiao, Hao Liu, Hongxuan Tang, Ying Chen,
  International Conference on Learning Representa-                 Hua Wu, and Haifeng Wang. 2022. A fine-grained
  tions, ICLR 2025, Singapore, April 24-28, 2025.                  interpretability evaluation benchmark for neural NLP.
  OpenReview.net.                                                  CoRR, abs/2205.11097.

Shauli Ravfogel, Michael Twiton, Yoav Goldberg, and             Yongjie Wang, Xiaoqi Qiu, Yu Yue, Xu Guo, Zhiwei
  Ryan Cotterell. 2022. Linear adversarial concept                Zeng, Yuhong Feng, and Zhiqi Shen. 2024. A sur-
  erasure. CoRR, abs/2201.12091.                                  vey on natural language counterfactual generation.
                                                                  In Findings of the Association for Computational
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:             Linguistics: EMNLP 2024, Miami, Florida, USA,
  Sentence embeddings using siamese bert-networks.                November 12-16, 2024, pages 4798–4818. Associa-
  In Proceedings of the 2019 Conference on Empiri-                tion for Computational Linguistics.
  cal Methods in Natural Language Processing and
  the 9th International Joint Conference on Natural             Zach Wood-Doughty, Ilya Shpitser, and Mark Dredze.
  Language Processing, EMNLP-IJCNLP 2019, Hong                    2018. Challenges of using text classifiers for causal
  Kong, China, November 3-7, 2019, pages 3980–3990.               inference. CoRR, abs/1810.00956.
  Association for Computational Linguistics.
                                                                Tongshuang Wu, Marco Túlio Ribeiro, Jeffrey Heer, and
Marco Túlio Ribeiro, Sameer Singh, and Carlos                     Daniel S. Weld. 2021. Polyjuice: Generating coun-
 Guestrin. 2016. "why should I trust you?": Ex-                   terfactuals for explaining, evaluating, and improving
 plaining the predictions of any classifier. CoRR,                models. In Proceedings of the 59th Annual Meeting
 abs/1602.04938.                                                  of the Association for Computational Linguistics and
                                                                  the 11th International Joint Conference on Natural
                                                                  Language Processing, ACL/IJCNLP 2021, (Volume 1:
Marcel Robeer, Floris Bex, and Ad Feelders. 2021. Gen-
                                                                  Long Papers), Virtual Event, August 1-6, 2021, pages
  erating realistic natural language counterfactuals. In
                                                                  6707–6723. Association for Computational Linguis-
 Findings of the Association for Computational Lin-
                                                                  tics.
  guistics: EMNLP 2021, Virtual Event / Punta Cana,
 Dominican Republic, 16-20 November, 2021, pages
                                                                Zhengxuan Wu, Karel D’Oosterlinck, Atticus Geiger,
  3611–3625. Association for Computational Linguis-
                                                                  Amir Zur, and Christopher Potts. 2022. Causal proxy
  tics.
                                                                  models for concept-based model explanations. CoRR,
                                                                  abs/2209.14279.
Wojciech Samek, Grégoire Montavon, Sebastian La-
 puschkin, Christopher J. Anders, and Klaus-Robert              Zhengxuan Wu, Atticus Geiger, Christopher Potts, and
 Müller. 2021. Explaining deep neural networks and                Noah D. Goodman. 2023. Interpretability at scale:
 beyond: A review of methods and applications. Proc.              Identifying causal mechanisms in alpaca. CoRR,
 IEEE, 109(3):247–278.                                            abs/2305.08809.
Pratinav Seth and Vinay Kumar Sankarapu. 2025.                  Fan Yang, Mengnan Du, and Xia Hu. 2019. Evaluat-
  Bridging the gap in xai-why reliable metrics mat-               ing explanation without ground truth in interpretable
  ter for explainability and compliance.     CoRR,                machine learning. CoRR, abs/1907.06831.
  abs/2502.04695.
                                                                Chih-Kuan Yeh, Been Kim, Sercan Ömer Arik, Chun-
Ruihao Shui, Yixin Cao, Xiang Wang, and Tat-Seng                  Liang Li, Tomas Pfister, and Pradeep Ravikumar.
  Chua. 2023. A comprehensive evaluation of large lan-            2020. On completeness-aware concept-based expla-
  guage models on legal judgment prediction. CoRR,                nations in deep neural networks. In Advances in
  abs/2310.11761.                                                 Neural Information Processing Systems 33: Annual
                                                                  Conference on Neural Information Processing Sys-
Qwen Team. 2023. Qwen: The official repo of qwen                  tems 2020, NeurIPS 2020, December 6-12, 2020,
 chat.                                                            virtual.

James Thorne,       Andreas Vlachos,      Christos              Wei Jie Yeo, Ranjan Satapathy, and Erik Cambria. 2024.
  Christodoulopoulos, and Arpit Mittal. 2019.                     Towards faithful natural language explanations: A
  Generating token-level explanations for natural                 study using activation patching in large language
  language inference. CoRR, abs/1904.10717.                       models. CoRR, abs/2410.14155.


                                                           14
```

### Page 15

![Rendered page 15](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-015.png>)

#### Extracted Text

```text
Xuemin Yu, Fahim Dalvi, Nadir Durrani, and Hassan               A     Discussion
  Sajjad. 2024. Latent concept-based explanation of
  NLP models. CoRR, abs/2404.12545.
                                                                A.1    Real-World Data
Raymond Zhang, Neha Nayak Kennard, Daniel Scott
  Smith, Daniel A. McFarland, Andrew McCallum,                  Why is it acceptable that the LIBERTy SCM
  and Katherine Keith. 2023. Causal matching with               does not perfectly reflect the real world? We do
  text embeddings: A case study in estimating the               not claim that our benchmark explains real-world
  causal effects of peer review policies. In Findings of        phenomena or reveals how LLMs internally repre-
  the Association for Computational Linguistics: ACL
  2023, Toronto, Canada, July 9-14, 2023, pages 1284–
                                                                sent them. Rather, our goal is to provide a princi-
  1297. Association for Computational Linguistics.              pled benchmark for comparing explanation meth-
                                                                ods, analyzing their limitations, and identifying
Haiyan Zhao, Hanjie Chen, Fan Yang, Ninghao Liu,                those that most faithfully capture model behavior,
  Huiqi Deng, Hengyi Cai, Shuaiqiang Wang, Dawei                thereby enabling their application in real-world set-
  Yin, and Mengnan Du. 2024. Explainability for large
  language models: A survey. ACM Trans. Intell. Syst.           tings. Therefore, it does not matter whether LIB-
  Technol., 15(2):20:1–20:38.                                   ERTy SCMs reflect real-world mechanisms (or are
                                                                just inspired by them). Explainability faithfulness
                                                                is defined with respect to the explained model, and
                                                                an explanation method should account for the ef-
                                                                fects of concepts as they are encoded by the model,
                                                                regardless of whether the model learns and repre-
Appendix                                                        sents real causal structures, synthetic structures, or
                                                                spurious correlations.

      A Discussion                              15
                                                                A.2    Deterministic Decoding
         A.1 Real-World Data . . . . . . . .    15
         A.2 Deterministic Decoding . . . .     15              Why is deterministic decoding necessary? While
         A.3 LLM-generated Counterfactuals      16              deterministic decoding has clear drawbacks, it is
                                                                essential for LIBERTy. Counterfactual generation
         A.4 Opportunities . . . . . . . . .    16
                                                                requires fixing the exogenous variables of the DGP.
                                                                Yet, stochastic decoding introduces noise at the
      B Human Validation                        16
                                                                token-sampling level that lies outside the DGP and
      C Explainability Methods                  17
                                                                cannot be controlled or recorded. As a result, such
                                                                counterfactuals cannot serve as ‘structural’ ones.
         C.1 Counterfactual Generation . . .    17
                                                                Furthermore, they also fail by intuitive standards.
         C.2 Matching . . . . . . . . . . . .   17              Although the prompt for generating the original ex-
         C.3 Concept Erasure . . . . . . . .    18              ample and the counterfactual may differ only in one
         C.4 Concept Attributions . . . . . .   18              concept value, stochastic decoding often produces
                                                                an entirely new narrative with little lexical over-
      D Dataset Details                         20              lap. While lexical overlap is not formally required,
                                                                it remains a widely used proxy for counterfactual
         D.1 Workplace Violence . . . . . .     20
                                                                quality in NLP. Accordingly, many works gener-
         D.2 Disease Detection . . . . . . .    23
                                                                ate counterfactuals by instructing LLMs to edit the
         D.3 CV Screening . . . . . . . . .     26              original text minimally (Gat et al., 2023; Li et al.,
                                                                2024; Wang et al., 2024). However, such examples
      E Implementation Details                  30              are only approximations, since entirely different
         E.1 Explainability Methods . . . .     30              DGPs produce the original and counterfactual texts.
         E.2 Explained Models . . . . . . .     30              Alternative solutions, beyond our approach of us-
                                                                ing exogenous grounding texts, include generat-
         E.3 Prompts . . . . . . . . . . . .    30
                                                                ing multiple counterfactuals and estimating ICaCE
      F Additional Results                      31
                                                                by averaging over them, or employing controlled
                                                                decoding methods for counterfactual generation
                                                                (Chatzi et al., 2025; Ravfogel et al., 2025).

                                                           15
```

### Page 16

![Rendered page 16](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-016.png>)

#### Extracted Text

```text
A.3   LLM-generated Counterfactuals                          fine-tuning, approaches that explicitly aim to align
                                                             models with the underlying DGP. To summarize,
Why do explanations based on LLM-generated
                                                             there is great promise in developing smaller, theory-
counterfactuals fail? Explanations based on
                                                             grounded, causal-inspired explainability and learn-
LLM-generated counterfactuals perform surpris-
                                                             ing approaches. We hope our work will serve as a
ingly well in benchmarks such as CEBaB (Gat
                                                             foundation for such future research.
et al., 2023), where human annotators provide the
reference counterfactuals against which explana-
tions are evaluated. However, this performance
stems from the fact that both humans and LLMs                B   Human Validation
approach the task similarly, by minimally editing
the input text to reflect a change in concept. In
                                                             We conduct human validation of the generated ex-
such settings, LLMs can closely mimic the refer-
                                                             amples to ensure: (1) they include all concept val-
ences, particularly when the texts are short and
                                                             ues; (2) they have high linguistic quality, by mea-
simple. Evaluation using human-written counter-
                                                             suring coherence and fluency; (3) they are relevant
factuals is therefore not an assessment of causal
                                                             to the task (e.g., look like a personal statement);
effects, but rather an evaluation of how well mod-
                                                             (4) they are logically consistent with themselves
els mimic human editing. When evaluated under
                                                             and external facts; (5) the counterfactual feels like
LIBERTy, however, the limitations of this approach
                                                             a genuine counterfactual (by measuring how likely
become clear. Unlike human-written counterfac-
                                                             the text was written by the same person in a par-
tuals, LIBERTy provides structural counterfactu-
                                                             allel world where the concept value is different).
als derived from causal interventions in the DGP.
                                                             Notably, human validation is not required to ensure
LLM-generated counterfactuals fail in this setting
                                                             that the LIBERTy evaluation pipeline is faithful;
because their edits reflect heuristic assumptions,
                                                             however, it helps demonstrate that the synthetically
rather than the actual underlying mechanism.
                                                             generated data is realistic and practical.
A.4   Opportunities                                             We recruited 13 annotators (all graduates with
What are the opportunities in the intersec-                  fluent English; 3 males, 10 females) who annotated
tion between causality, explainability, and                  a total of 349 single-text and 312 text-cf-pair eval-
NLP/LLMs? Our findings reveal several exciting               uations. Each text was rated across six dimensions:
opportunities at the intersection of causality, ex-          five individual attributes assessing text-level quality
plainability, and NLP. First, we observed a large            and one comparative attribute assessing the quality
room for improvement in both local and global ex-            of the counterfactual relative to its original. This
planations, offering clear targets for future work.          resulted in a total of 349 × 5 + 312 × 1 = 2,057 la-
There is clear potential for the development of              bels. The average inter-annotator agreement (IAA)
causal-inspired explanation methods. Instead of re-          across all dimensions is 0.91. The annotation guide-
lying on LLM-based explanations, which, despite              lines can be viewed in Figures 4 and 5.
encoding broad knowledge in their parameters, are               The results are presented in Table 5. As shown,
not exposed to data from the target DGP and there-           the generated examples exhibit high linguistic qual-
fore fail to provide faithful explanations, small but        ity, with average scores of 4.79 and 4.85 out of 5 for
principled techniques offer a more promising di-             coherence and fluency, respectively. Their average
rection. These approaches can rely on causal struc-          scores for task relevance and logical consistency
ture rather than scale, making them especially well-         are 4.77 and 4.92. In addition, agreement with con-
suited for academic research. LIBERTy provides a             cept values is 94.2% on average, indicating that
rigorous evaluation ground for such methods and,             GPT-4o accurately instantiates the sampled values.
we hope, will foster their further development.              The lowest scores appeared in the CV Screening
   Finally, our analysis revealed that vanilla fine-         dataset, probably because it involves socially sensi-
tuning may fail to capture the causal structure of           tive concepts that are more heavily filtered during
the data, suggesting the need for unique learning            generation. Finally, annotators judged the counter-
methods. This opens an opportunity to harness LIB-           factuals to be genuine, with an average score of
ERTy as a testbed for developing and benchmark-              4.44 out of 5, demonstrating that they were per-
ing new causal learning methods that go beyond               ceived as plausible even by the human eye.

                                                        16
```

### Page 17

![Rendered page 17](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-017.png>)

#### Extracted Text

```text
                                         Workplace       Disease         CV
                                                                                     Avg.
                                          Violence      Detection     Screening
                        # Annotators          6               5           6          5.67
                        # Individual         76              170         103        116.33
                        # Pairs              101             105         106         104
                        # Labels             481             955         621        685.67
                        Avg. IAA             0.90          0.92          0.91        0.91
                        Avg. MAE             0.35          0.53          0.62        0.50
                        Concepts            97.9%         100%          84.7%       94.2%
                        Coherence            4.75          4.88          4.75        4.79
                        Fluency              4.72          4.90          4.92        4.85
                        Relevancy            4.80          4.68          4.83        4.77
                        Consistency          4.92          4.92          4.92        4.92
                        Plausibility         4.63          4.62          4.07        4.44

Table 5: Results of Human Validation: Average IAA and MAE are computed across annotator pairs: IAA for
the binary concept identification task, and MAE for all other tasks using a 1–5 Likert scale. ‘Concepts’ reports the
percentage of concept values that were marked as explicitly stated or logically inferred. ‘Plausibility’ reports the
average score for a pair of texts being judged as an original and its counterfactual.


C     Explainability Methods                                 structs the LLM to fix the values of all other con-
                                                             cepts; (c) Fix Confounders: specifies the change
In this section, we provide additional background            and the causal parents, explicitly forbidding their
on the explainability methods used in our study, as          alteration.; (d) Mediators and Confounders: spec-
well as further implementation details for each.             ifies all mediator concepts (without asking to fix
                                                             their values) and the change, while instructing the
C.1    Counterfactual Generation
                                                             LLM to fix the values of the confounding concepts.
This approach uses an LLM (or a fine-tuned, pre-                To generate counterfactuals, we use Gemini-1.5-
trained model when parallel training data are avail-         Pro, which differs from the LLM used to gener-
able) to generate approximations of counterfactuals.         ate LIBERTy examples (GPT-4o). Importantly,
Typically, the LLM is instructed to modify the in-           although the prompts may mention the concepts
put text by replacing a specified concept with a             and sometimes their roles (confounders or media-
target value. Gat et al. (2023) propose injecting            tors), Gemini is expected to infer on its own how
causal assumptions into the prompt, in particular            a change in the target concept affects other con-
identifying confounder concepts from the causal              cepts (if they are mediators) and the resulting text.
graph and prompting the LLM to keep them fixed               To compare different prompting techniques and
while changing the target concept. They found that           manage computational costs, we restrict our ex-
LLM-generated counterfactuals yielded the best               periments to the CV Screening dataset and three
explanation method on CEBaB. In light of this,               fine-tuned models: DeBERTa-base, T5-base, and
we extend their approach and compare different               Qwen2.5-1.5B. The results are reported in Table 6.
prompting strategies, each of which injects distinct         As shown, the best-performing prompting tech-
causal assumptions into the prompt. In our causal            nique is Mediators and Confounders, which is also
graphs, relative to the target concept being modi-           the most causally informed. This technique explic-
fied, other concepts may play two key roles. The             itly incorporates both causal roles: it asks to hold
first are confounders, which act as root causes that         the confounders fixed while allowing mediators to
influence both the target concept and the text, and          vary according to Gemini’s decision. Since this
therefore must remain fixed. The second are media-           technique works the best, we use it in all other ex-
tors, which are influenced by the target concept and,        periments. The full set of prompt versions used for
in turn, influence the text. They must be allowed to         this task is provided in Appendix E.3.2.
vary when measuring total causal effects.
   The prompting techniques we evaluate are: (a)             C.2    Matching
Only Change: specifies only the target concept               Although counterfactual generation is a valuable
change; (b) Fix All: specifies the change and in-            explainability approach, employing LLMs during

                                                        17
```

### Page 18

![Rendered page 18](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-018.png>)

#### Extracted Text

```text
                     → Model            Average      DeBERTa-v3              T5          Qwen-2.5
                     ↓ Technique       ED    OF      ED    OF          ED         OF     ED   OF
                     Only Change       0.59   0.49   0.54     0.51    0.50        0.50   0.72   0.46
                     Fix All           0.54   0.58   0.46     0.62    0.44        0.61   0.72   0.50
                     Fix Confounders   0.55   0.55   0.49     0.57    0.46        0.58   0.71   0.50
                     Meds & Confs      0.57   0.54   0.48     0.58    0.49        0.55   0.73   0.48

Table 6: Results of Counterfactual Generation Prompting: We report the Average Error Distance (ED) and
Average Order-Faithfulness (OF) for the four prompting techniques used in counterfactual generation with Gemini-
1.5-Pro. Meds & Confs is Mediators and Confounders: mentioning mediators while instructing to fix confounders.


inference can be costly, either due to latency or            vector and compute cosine similarity between this
financial expenses. An alternative is to use a more          vector for the original example and each candidate.
efficient method that searches for approximations
within a predefined set of candidate texts. This             C.3     Concept Erasure
approach, known as matching, involves identifying            Concept erasure methods intervene on a model’s in-
the most similar candidate text whose target con-            ternal representations to remove information about
cept corresponds to the desired target value. Match-         a target concept, typically by projecting out direc-
ing methods differ in how they perform the search.           tions in the activation space that encode it. By
We evaluate two approaches: matching based on                comparing model behavior before and after era-
semantic similarity and matching based on concept            sure, these methods estimate the influence of the
values. A third approach involves learning causal            concept on predictions. In this study, we evaluate
representations (Gat et al., 2023), which lies out-          the state-of-the-art erasure method LEACE (Bel-
side the scope of our study. In addition, we adopt           rose et al., 2023). LEACE is a closed-form method
the top-k matching technique (with k = 3), which             that removes all linearly encoded information about
has been shown to outperform single matching (Gat            a target concept, while minimizing distortion to
et al., 2023).                                               other directions. Given a hidden representation
                                                             h(x) LEACE computes an affine projection that
Semantic-based Matching For each original                    eliminates the components aligned with the con-
text and concept change C : c → c′ , we retrieve             cept direction vc . This yields an erased representa-
the top-k candidates with C = c′ based on co-                tion herased-c (x). The effect of the concept is then
sine similarity between mean-pooled text embed-              defined as the difference between the model’s pre-
dings. To compute embeddings, we examine three               dictions for h(x) and on herased-c (x).
encoder-only models: (1) ST Match: a Sentence-                  Applicability Note: In our experiments, we ap-
Transformer model (the default ‘all-MiniLM-L6-               ply LEACE by extracting embeddings via mean
v2’ model) (Reimers and Gurevych, 2019); (2) PT              pooling. Since LEACE assumes that a concept
Match: a pre-trained DeBERTa model (DeBERTa-                 value of 0 corresponds to the concept being absent,
base version); and (3) FT Match: a DeBERTa                   we restrict its use to the D ISEASE D ETECTION
model fine-tuned to predict Y in each dataset.               dataset, where this assumption holds naturally (e.g.,
                                                             symptom absence). In other datasets, the concepts
Concept-based Matching For each original text                of interest involve changes between two non-null
and concept change C : c → c′ , we retrieve the              states (e.g., gender, occupation), for which the “ab-
top-k candidates with C = c′ based on similarity             sence” assumption does not apply, making erasure
of the remaining concept values. Since we assume             ill-defined. Finally, because LEACE requires ac-
that the explanation method does not have direct             cess to and modification of internal embeddings,
access to the gold concept labels, we fine-tune a De-        we apply it only to fine-tuned models that sup-
BERTa model (DeBERTa-base version) to predict                port this interface: DeBERTa-base, T5-base, and
concept values from text. Matching is then per-              Qwen2.5-1.5B in our evaluation.
formed in two alternative ways: (1) Approx — all
other concept values must match exactly, with a sin-         C.4     Concept Attributions
gle mismatch permitted only if no perfect match is           Concept attribution methods map concepts to vec-
available; (2) ConVecs — we concatenate the soft-            tors or subspaces within a model’s internal activa-
max prediction vectors of all concepts into a single         tion space, typically derived from concept-labeled

                                                        18
```

### Page 19

![Rendered page 19](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-019.png>)

#### Extracted Text

```text
examples. These vectors capture directions in the
hidden representation space that the model relies on
for prediction, enabling us to quantify how move-
ment along a concept direction affects the model’s
output and thereby assess the concept’s importance.
In our experiments, we combine ConceptShap (Yeh
et al., 2020) with TCAV (Kim et al., 2018), two
widely used concept attribution methods in com-
puter vision. ConceptShap quantifies the contri-
bution of concepts to a model’s predictive perfor-
mance using Shapley values. Unlike TCAV, which
measures directional sensitivity along a single con-
cept vector, ConceptShap treats concepts as play-
ers in a cooperative game and attributes credit to
them based on their marginal contributions across
all possible coalitions of concepts. To apply this
framework, one first requires a representation for
each concept and then computes Shapley values.
Since our goal is to evaluate predefined concepts,
we construct their representations using TCAV vec-
tors. TCAV derives concept vectors by training a
linear classifier in the activation space to separate
examples that contain the concept from those that
do not, and then uses the classifier’s normal vec-
tor as the concept representation. ConceptShap is
then applied over these predefined vectors to assign
Shapley-based importance scores.
   Applicability Note: Both ConceptShap and
TCAV are primarily global explanation methods:
they quantify how concepts influence the model’s
predictions across a dataset, rather than for individ-
ual inputs. Accordingly, we evaluate them only in
the global explainability setup.




                                                         19
```

### Page 20

![Rendered page 20](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-020.png>)

#### Extracted Text

```text
D       Dataset Details


D.1      Workplace Violence


D.1.1     SCM

This dataset simulates HR–nurse interviews, in which the (explained) model predicts the likelihood that
a nurse will experience workplace violence. The causal graph is adapted from the Minnesota Nurses’
Study (Gerberich et al., 2004), which documented the prevalence of verbal and physical violence among
clinical staff and analyzed risk factors by demographic and professional background. We perform minor
simplifications to reduce the number of concepts and to rename them for clarity. The simplified version
preserves the main causal relations reported in the original paper while maintaining readability.
   The template follows a structured HR interview format. To ensure both realism and sufficient diversity,
we generate interview templates as follows: for each concept, a bank of 10 questions is created using
Gemini, each designed to elicit the concept’s value from different linguistic perspectives. Additionally, 10
opening and 10 closing sentence variants are defined to maintain a coherent interview flow. Each template
is generated by sampling one question per concept, along with an opening and closing sentence. The
question order is randomized, yielding a large pool of interview templates. The persona contains three
informal “fun facts” about the nurse, each centered on a concept (without specifying its value). Using
Gemini, we generated 500 personas.



    C      Name                                       Values                         Parents        Childs

                                      {0: No Violence, 1: Verbal Violence, 2:
    Y      Violence Experience                                                          all            –
                                                Physical Violence}
    G      Gender                              {0: Female, 1: Male}                     –            L, D
    A      Age                            {0: 24–32, 1: 34–44, 2: 46–55}                –            T, L
                                       {0: African American, 1: Hispanic, 2:
    R      Race                                                                         –          L, D, S, Y
                                                 White, 3: Asian}
    T      Tenur                           {0: 4–9, 1: 10–19, 2: 20–25}                 A            S, Y
    L      License                          {0: LPN, 1: RN, 2: APRN}                 G, R, A         S, Y
                                          {0: Family Practice, 1: ICU, 2:
    D      Department                      Psychiatric/Mental Health, 3:               G, R            Y
                                                   Emergency}
                                     {0: General Staff, 1: Experienced Staff, 2:
    S      Seniority                      Middle Management, 3: Senior             A, G, R, T, L       Y
                                                   Management}

    G ∼ Uniform{0, 1}
    A ∼ Categorical{0: 25%, 1: 50%, 2: 25%}
    R ∼ Uniform{0, 1, 2, 3}
    T = min(2, max(0, round(0.8 A + εT )))    εT ∼ N (0.05, 0.5)
    L = min(2, max(0, round(0.3 G + 0.3 R + 0.2 A + εL )))    εL ∼ N (0, 0.5)
    D = min(3, max(0, round(0.5 G + 0.4 R + 0.4 + εD )))    εD ∼ N (0.2, 0.5)
    S = min(3, max(0, round(0.4 A + 0.1 (G + R) + 0.3 (T + L) + εS )))    εS ∼ N (0, 0.5)
    Y = min(2, max(0, round(0.5 (G + D) − 0.2 (A + R + L + T + S) + 0.8 + εY )))      εY ∼ N (0.3, 0.2)

                          Table 7: SCM of the Workplace Violence Prediction Dataset.




                                                        20
```

### Page 21

![Rendered page 21](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-021.png>)

#### Extracted Text

```text
D.1.2   Prompts
   Box D.1: Nurse Persona Generation Prompt
  System Instruction:
  Your task is to create an engaging nurse persona by generating fun facts for three given aspects. These
  facts should highlight the nurse’s professional or personal journey.
  User Prompt:
  Here     are     the     three    aspects:        {sample_aspects[0]},             {sample_aspects[1]},
  {sample_aspects[2]}.
  Please creatively generate three surprising and contextually relevant fun facts for each aspect that highlight
  the nurse’s professional or personal journey.
  Aim to enrich the persona and captivate the audience by revealing unique insights into the nurse’s
  experiences.
  Respond in this format:
  Fun Fact on {sample_aspects[0]}:
  Fun Fact on {sample_aspects[1]}:
  Fun Fact on {sample_aspects[2]}:


   Box D.2: Original & Counterfactual Nurse Dialogue Generation Prompt
   System Instruction:
   As a specialist in refining dialogues between HR personnel and a nurse, your task is to enhance the
   conversation with added depth, personal insights, and storytelling. The primary goal is to remain fully
   consistent with the nurse’s personal information provided. You will also be given fun facts about the
   nurse’s persona. Use these to enrich the dialogue, but adjust the facts as needed to ensure they align
   with the personal information. If any fun fact conflicts with the personal information, rewrite it to match.
   Finally, make sure the resulting dialogue feels coherent and natural. Avoid repeating questions or asking
   something that has already been mentioned. Ensure that everything flows smoothly, as if it were a real and
   authentic conversation.
   User Prompt:
   Based on the provided base dialogue, revise the conversation to incorporate more depth and include all
   adjusted fun facts from the nurse’s persona. Ensure these fun facts align with the nurse’s personal
   information; revise any discrepancies to accurately reflect the nurse’s true values.

   Nurse’s personal information: {nurse_details}
   Nurse’s Persona: {nurses_persona}
   Base dialogue: {dialogue_draft}

   Final dialogue:


D.1.3   Examples
   Box D.3: Example of Nurse Dialogue Template
   Intro: Excited for our chat. I’m from HR, and we’ve got a brief 5-minute discussion ahead to collect
  some personal and demographic information. How have you been coping with everything?
  Department Question: Just for clarity, can you tell us your specific department?
  Department Info: Intensive Care Unit (ICU)
  Race Question: How would you describe your race or ethnicity?
  Race Info: African American
  Age Question: How old are you, if you’re comfortable sharing?
  Age Info: 44
  Gender Question: Just to get a clearer picture, could you tell me your gender?
  Gender Info: Male
  License Type Question: Could you indicate which nursing license you’ve obtained? LPN, RN, or
  APRN?
  License Type Info: Registered Nurse (RN)
  Years As Nurse Question: Could you let us know how long you’ve been working in the nursing field?
  Years As Nurse Info: 10
  Activity At Work Question: Could you describe the extent of decision-making authority you hold in
  your current role?
  Activity At Work Info: Experienced Staff
  Closing: Short chat, but it was packed with insights. Thanks for helping me out. Bye for now.




                                                       21
```

### Page 22

![Rendered page 22](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-022.png>)

#### Extracted Text

```text
Box D.4: Example of Generated Nurse Persona
  Fun Fact on Race: While studying to become a nurse, she volunteered at a free clinic that primarily
served a community different from her own, opening her eyes to healthcare disparities and solidifying her
commitment to equitable care.
Fun Fact on Age: She decided to become a nurse later in life, after a successful career in a completely
unrelated field, bringing a unique perspective and wealth of life experience to her nursing practice.
Fun Fact on Department: Although she always envisioned herself working in a fast-paced environment
like the ER, she discovered a passion for palliative care, finding deep fulfillment in providing comfort and
support to patients and their families during challenging times.



Box D.5: Example of Original and Counterfactual Generated Nurse Dialogue
Changed aspect: Gender from Male to Female and Department from “Psychiatric/Mental Health” to
“Intensive Care Unit (ICU)”.

Original Dialogue:
HR: Hi James, I’m excited for our chat today. I’m from HR, and we’ve got a brief 5-minute discussion
ahead to collect some personal and demographic information. How have you been coping with everything?
James: Hi! I’ve been doing well, thank you. It’s been a busy time, but I’m managing to keep everything
balanced.
HR: That’s good to hear. Let’s dive in. Could you tell me about your department? What challenges and
rewards come with working there?
James: I work in the Psychiatric/Mental Health department. It’s a challenging field, but incredibly
rewarding. Every day, I get to make a difference in people’s lives, helping them navigate their mental
health journeys. It’s a privilege to support them through their toughest times.
HR: That sounds like meaningful work. Just to get a clearer picture, could you tell me your gender?
James: Sure, I’m male.
HR: And would it be okay to tell me your age?
James: I’m 43 years old. Turning 40 was a milestone for me, and it inspired me to fulfill a lifelong dream
of learning Spanish. It’s been incredibly rewarding to connect with my Hispanic patients on a deeper
level.
HR: That’s wonderful! How would you describe your race or ethnicity?
James: I’m African American. My heritage is an important part of who I am, and I strive to bring that
perspective into my work every day.
HR: Thank you for sharing that. I’m interested in knowing, how many years have you been nursing?
James: I’ve been a nurse for 10 years now. It’s been a fulfilling journey, and I continue to learn and grow
every day.
HR: Could you tell us whether you are an LPN, RN, or APRN?
James: I’m a Licensed Practical Nurse (LPN). I initially trained as an RN, but I found my calling as an
LPN, where I can have a direct impact on patient care.
HR: Could you describe the extent of decision-making authority you hold in your current role?
James: As an experienced staff member, I have a significant role in decision-making, especially when it
comes to patient care plans. I work closely with the team to ensure we’re providing the best possible
support for our patients.
HR: It sounds like you have a lot of responsibility and passion for your work. Short chat, but it was
packed with insights. Thanks for helping me out. Bye for now.
James: Thank you! It was great talking to you.


CF Dialogue:
HR: Hi, I’m excited for our chat today. I’m from HR, and we’ve got a brief 5-minute discussion ahead to
collect some personal and demographic information. How have you been coping with everything?
Nurse: Hi there! I’ve been managing well, thank you. The ICU can be intense, but it’s also incredibly
rewarding.
HR: I can imagine. Tell me about your department. What challenges and rewards come with working
there?
Nurse: I work in the Intensive Care Unit (ICU). It’s a challenging environment because we deal with
critically ill patients, but it’s rewarding to see them recover and know that I played a part in their
journey. The teamwork here is phenomenal, and we support each other through the toughest days.
HR: That sounds like a very fulfilling role. Just to get a clearer picture, could you tell me your gender?
Nurse: Sure, I’m female.
HR: Would it be okay to tell me your age?
Nurse: I’m 43 years old. It’s been an interesting journey, especially since turning 40, which inspired me
to start learning Spanish. It’s been wonderful to connect with my Hispanic patients on a deeper level.
HR: That’s impressive! How many years have you been nursing?
Nurse: I’ve been a nurse for 10 years now. I started my career as an LPN, and it’s been a fulfilling



                                                    22
```

### Page 23

![Rendered page 23](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-023.png>)

#### Extracted Text

```text
   decade of learning and growth.
   HR: How would you describe your race or ethnicity?
   Nurse: I’m African American. My heritage is an important part of who I am, and I strive to bring that
   perspective into my work every day.
   HR: Could you describe the extent of decision-making authority you hold in your current role?
   Nurse: As an experienced staff member, I have a significant amount of decision-making authority. I often
   collaborate with doctors and other nurses to determine the best care plans for our patients. It’s a role
   that requires both leadership and teamwork.
   HR: Could you tell us whether you are an LPN, RN, or APRN?
   Nurse: I’m a Licensed Practical Nurse (LPN). I initially trained as an LPN because I wanted to get into
   the field quickly and start making a difference. It’s been a rewarding path, and I continue to learn every
   day.
   HR: Short chat, but it was packed with insights. Thanks for helping me out. Bye for now.
   Nurse: Thank you! It was great talking to you. Have a wonderful day!



D.2     Disease Detection
D.2.1     SCM
This dataset simulates clinical self-reports, where the (explained) model predicts a disease from symptoms
described in a medical forum post. Unlike the other two datasets, the learning problem is anti-causal:
the disease label serves as the root cause in the SCM and determines the values of symptom concepts,
based on known symptom–disease relations (Monto et al., 2000; Cady and Schreiber, 2002). We also used
we used domain knowledge from the Cleveland Clinic10 to identify the key symptoms associated with
each condition. Each disease node serves as a parent node to its characteristic symptoms, some of which
overlap across diseases to introduce realistic confounding. Dependencies between symptoms (e.g., bright
light affecting headache) were explicitly modeled as causal edges. Additionally, symptom prevalence was
modeled in the SCM functions, such that more characteristic symptoms have stronger causal weights (e.g.,
facial pain is more likely than fever for sinusitis).
   The template is a narrative structure abstracted from 1,310 posts on Reddit’s DiagnoseMe forum,11
using Gemini to preserve the clinical tone and flow. The persona (a total of 1200) consists of three
informal facts about occupation, hobbies, and family or friends. To generate personas, we first sample an
occupation and a hobby from predefined lists, then use Gemini to generate the corresponding facts. Each
dataset example is created by prompting GPT-4o to follow the template and integrate information from
the persona and the symptom values.

D.2.2     Prompts
   Box D.6: Disease Template Generation Prompt
   System Instruction:
   "Develop a narrative template based on the structure of the provided example. The template should abstract
   the formatting and key transitions from the example, while seamlessly integrating occupation and hobby
   details into the narrative. Use this template to ensure that any future persona creation maintains the
   coherence and style of the original example, yet allows for flexibility to adapt to different personas and
   symptoms."
   User Prompt:
   **Analyze Example Format**: {reddit_comment}
   From the example provided, analyze and extract the fundamental structure and style used in composing the
   narrative:
   1. Analyze Example Format: Focus on how the example is constructed, noting key phrases, transitions,
   the arrangement of topics, and how personal details are woven into the narrative.
   2. Craft a Template: Using your analysis, create a narrative template that includes placeholders or
   cues for integrating occupation and hobby. Ensure the template can be easily adapted to different scenarios
   while maintaining the style and coherence of the example.
   Your Task: Generate a narrative template that can be used to create engaging and coherent personas
   based on any set of personal details, following the style and structure of the example provided.


  10
       https://my.clevelandclinic.org/health/diseases
  11
       https://www.reddit.com/r/DiagnoseMe/


                                                       23
```

### Page 24

![Rendered page 24](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-024.png>)

#### Extracted Text

```text
C      Name                                        Values                        Parents           Childs

Y      Disease                     {0: Migraine, 1: Sinusitis, 2: Influenza}        –                all
D      Dizziness                       {0: Absent, 1: Mild, 2: Strong}              Y                 –
L      Light Sensitivity               {0: Absent, 1: Mild, 2: Strong}              Y                H
P      Facial Pain                     {0: Absent, 1: Mild, 2: Strong}              Y                 –
W      Weakness                        {0: Absent, 1: Mild, 2: Strong}              Y                 –
F      Fever                           {0: Absent, 1: Mild, 2: Strong}              Y                 –
N      Nasal Congestion                {0: Absent, 1: Mild, 2: Strong}              Y                H
H      Headache                        {0: Absent, 1: Mild, 2: Strong}           Y, L, N              –

Y = εY , εY ∼ Cat({0 : 13 , 1 : 13 , 2 : 31 })
D = min(2, max(0, round(0.9 · 1{Y = 0} + εD ))), εD ∼ N (−0.1, 0.6)
L = min(2, max(0, round(0.9 · 1{Y = 0} + εL ))), εL ∼ N (0.2, 0.5)
N = min(2, max(0, round(0.7 · 1{Y = 1} + 0.4 · 1{Y = 2} + εN ))), εN ∼ N (0, 0.7)
P = min(2, max(0, round(0.8 · 1{Y = 1} + εP ))), εP ∼ N (0.2, 0.6)
F = min(2, max(0, round(0.4 · 1{Y = 1} + 0.6 · 1{Y = 2} + εF ))), εF ∼ N (0, 0.6)
W = min(2, max(0, round(0.7 · 1{Y = 2} + εW ))), εW ∼ N (0.2, 0.6)
H = min(2, max(0, round(0.7 · 1{Y = 0} + 0.4 · 1{Y = 1} + 0.3L + 0.3N + εH ))), εH ∼ N (−0.1, 0.5)

                              Table 8: SCM of the Disease Detection Dataset.


 Box D.7: Disease Persona Generation Prompt
 System Instruction:
 "Your task is to create an engaging persona by generating three interesting facts covering their occupation,
 hobby, and personal life, based on the provided hobby and disease context."
User Prompt:
Create an engaging persona using the provided details:
Persona’s occupation: {occupation}
Persona’s hobby: {hobby}
** Respond in this format **:
Occupation: Detail the persona’s job and an interesting related fact/story. 1-2 sentences.
Hobby: Describe the persona’s hobby and how it enriches their life. 1-2 sentences.
Family/Friends: Share a brief story or fact about the persona’s interactions with family or friends. 1-2
sentences.


 Box D.8: Original & Counterfactual Disease Text Generation Prompt
System Prompt:
You are an AI assistant tasked with crafting a detailed consultation post for a patient seeking online
medical advice. The consultation should be developed by integrating the patient’s provided symptoms,
tailored persona details, and the structural guidance provided by the narrative template. It is essential
to explicitly incorporate each symptom and aspect of the patient’s personal background into the post.
Your goal is to create a ready-to-submit, engaging, and clear consultation request that effectively and
compellingly explains the patient’s situation.

 User Prompt:
 Compose an engaging and detailed consultation post using the following elements:
 1. Narrative Template: Use the provided template as a guiding framework to structure your consultation.
 It should shape the flow and organization of the post, ensuring a logical presentation of your symptoms
 and background story.
 2. Patient’s Symptoms List: This is the most crucial component—it includes the patient’s symptoms,
 which should be described in detail, focusing on their impact on daily activities and overall well-being.
 3. Persona Details: Enhance the narrative by incorporating persona details, such as lifestyle, hobbies,
 and family context, to give depth to the post. Explain how the symptoms affect specific aspects of the
 patient’s life.
 Narrative Template: {reddit_template}
 Patient’s Symptoms List: {verbal_symptoms_list}



                                                      24
```

### Page 25

![Rendered page 25](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-025.png>)

#### Extracted Text

```text
   Persona Details: {persona_info}
   Please ensure that the final output is a cohesive and engaging narrative without distinct section breaks.
   It should be medically informative and follow a logical flow, starting with an introduction that captures the
   reader’s attention,
   clearly explaining the symptoms and their impact, and concluding with a request for advice or further
   action.

D.2.3   Examples
   Box D.9: Example of Generated Disease Narrative Template
  Narrative Template for Persona Creation:
  1. Opening Statement (Expressing Frustration & Seeking Help):
  “I know this might be a lot, but [briefly explain the challenge of summarizing your symptoms, e.g., they
  feel scattered, doctors haven’t found a solution yet]. It’s been incredibly difficult to figure out where to even
  begin, and I’m feeling incredibly [emotion, e.g., overwhelmed, hopeless, lost]. The doctors I’ve seen have
  mainly focused on treating individual symptoms without getting to the root of the problem. I’m desperate
  for answers and wondering if there are any tests or specialists you could recommend.”
  2. Known Medical History (Concise & Factual):
  Existing Conditions: [List diagnosed conditions, including year of diagnosis if relevant].
  Current Medications: [List medications, dosage, and what they are taken for].
   3. Lifestyle (Brief & Relevant):
   Briefly describe lifestyle factors that could be relevant to health, e.g., smoking, alcohol consumption, diet].


   Box D.10: Example of Generated Disease Persona
   Occupation: As an Occupational Health and Safety Technician, they ensure workplaces are safe for
   everyone. They once investigated a case where someone nearly got stuck in a tunnel, highlighting the
   importance of their job.
   Hobby: Building tunnels as a hobby lets them apply their professional knowledge in a fun, challenging
   way. Plus, it’s incredibly satisfying to create underground spaces.
   Family/Friends: Their friends often joke about needing hard hats and safety briefings before visiting,
   but secretly, they’re fascinated by their hobby.


   Box D.11: Example of Original and Counterfactual Disease text
  Changed aspect: Remove: General_Weakness (Slight)

   Original disease’s symptom list: [Dizzy (Slight), Sensitivity_to_Light (Slight), Headache (Slight), Fever
   (Strong), General_Weakness (Slight)]

   Original disease text:
   Hi, I’m a 34-year-old Caucasian female living in Portland, Oregon. Lately, I’ve been struggling with
   feeling slightly off-balance, and it’s been really difficult to maintain my usual pace at work and enjoy my
   hobbies. This all started about three weeks ago, although thinking back, it might have been creeping up
   on me for a while. As a psychiatric technician, my role involves providing compassionate care and support
   to individuals facing mental health challenges. However, this minor instability, especially when moving
   quickly, has made it challenging to perform my duties with the same confidence and energy. Additionally,
   I’ve been feeling slightly fatigued, more tired than usual, which further complicates my ability to be fully
   present and engaged with my patients.
   The symptoms don’t stop there. I’ve noticed a strain in bright light, a subtle discomfort in my eyes
   when I’m in well-lit areas, which makes it hard to focus during meetings or when documenting patient
   progress. On top of that, I’ve been experiencing intense feverishness, feeling extremely warm and very
   ill at times, which is quite alarming. A mild headache, a gentle but persistent discomfort, accompanies
   these symptoms, making it difficult to concentrate on tasks or enjoy my favorite pastime, birdwatching.
   Birdwatching has always been my peaceful escape, where I find solace in identifying different species and
   often rise before dawn to catch a glimpse of rare birds. However, these symptoms have made it hard to
   enjoy these early morning adventures, especially when I share them with my niece and nephew, who I
   love introducing to the wonders of nature.
   I’m increasingly concerned about how these symptoms are impacting my daily life and my ability to
   create special memories with my family. Any advice would be greatly appreciated. Please assist me in
   returning to the vibrant life I cherish—filled with meaningful work, the joy of birdwatching, and precious
   moments with my loved ones.



                                                         25
```

### Page 26

![Rendered page 26](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-026.png>)

#### Extracted Text

```text
   Counterfactual disease’s symptom list: [Dizzy (Slight), Sensitivity_to_Light (Slight), Headache (Slight),
   Fever (Strong)]

   Counterfactual disease text:
   Hi, I’m a 34-year-old Caucasian female living in Portland, Oregon. Lately, I’ve been struggling with
   feeling slightly off-balance, and it’s been really difficult to maintain my usual routine, especially at work
   and during my cherished hobbies. This all started about three weeks ago, although thinking back, it might
   have been creeping up on me for a while.
   As a psychiatric technician, my role involves providing compassionate care and support to individuals
   facing mental health challenges. I find deep fulfillment in helping patients develop coping mechanisms
   and life skills. However, this minor instability, especially when moving quickly, has made it challenging to
   perform my duties effectively. I often feel a subtle discomfort in my eyes in well-lit areas, which adds to
   the strain during my shifts. The intense feverishness I experience makes me feel extremely warm and very
   ill, further complicating my ability to focus and be present for my patients. Additionally, a mild headache
   lingers, a gentle but persistent discomfort that seems to accompany me throughout the day.
   Outside of work, birdwatching has always been my peaceful escape. I love rising before dawn to catch
   a glimpse of rare birds, finding solace in identifying different species. However, the slight off-balance
   feeling and the strain in bright light have made these early morning excursions less enjoyable and more
   challenging. I also cherish sharing this passion with my niece and nephew, creating special memories on
   nature walks and fostering a love for the natural world. Yet, the symptoms have made it difficult to keep
   up with their youthful energy and enthusiasm.
   I’m reaching out for advice because these symptoms are increasingly impacting my daily life and the
   activities I hold dear. Any guidance or suggestions would be greatly appreciated. Please assist me in
   returning to the vibrant life I cherish—filled with meaningful work, peaceful birdwatching, and joyful
   moments with my family.


D.3     CV Screening
D.3.1     SCM
This dataset simulates automated resume assessment, where the model is tasked with predicting an appli-
cant’s quality from a CV-style personal statement, with labels such as weak, qualified, and outstanding.
Motivated by critiques of real-world screening systems (Dastin, 2018; Raghavan et al., 2020; Cowgill
et al., 2020), the causal graph encodes hypothesized dependencies between demographic and professional
attributes, inspired by statistical patterns reported by the U.S. Bureau of Labor Statistics.12 For example,
gender influences the hiring label only indirectly through mediators such as education and Work Experi-
ence. We examined multiple demographic and behavioral graphs to infer general causal tendencies, such
as differences in education continuation or volunteering rates across demographic groups.
   1,235 templates were generated from 342 scraped personal statement examples,13 where each source
text was abstracted with Gemini using a 2-shot prompt to produce several occupation-agnostic variants
that preserve the narrative structure while removing concept- and role-specific details. To generate a
persona (a total of 990), we sample a role from a predefined list and use Gemini with a 2-shot prompt to
produce both personal and professional context, including motivations and skills relevant to that role. Each
dataset example is then created by prompting GPT-4o to follow the template and integrate information
from the application role, the persona, and the sampled concept values.

D.3.2     Prompts
   Box D.12: CV Template Generation Prompt
   System Instruction:
   Create a short CV narrative template from the given personal statement example, distilling its essential
   structure and style. The template should include key transitions and be concise yet comprehensive, ensuring
   it can adapt to a variety of professional and personal profiles while preserving coherence and flexibility.
   User Prompt:
   Analyze Personal Statement: sampled_statement
   From the personal statement provided, analyze and extract the fundamental structure and style:
   1. Structure Analysis: Note key phrases, transitions, and arrangement of professional and personal
   information.

  12
       https://www.bls.gov/cps/demographics.htm
  13
       https://universitycompare.com


                                                        26
```

### Page 27

![Rendered page 27](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-027.png>)

#### Extracted Text

```text
C      Name                                         Values                       Parents            Childs

G      Gender                                {0: Female, 1: Male}                    –                E
                                      {0: Black, 1: Hispanic, 2: White, 3:
R      Race                                                                          –                E
                                                     Asian}
A      Age Group                        {0: 24–32, 1: 33–44, 2: 45–55}               –             E, S, W
                                       {0: High School, 1: Bachelor’s, 2:
E      Education                                                                  G, R, A        S, W, V, C, Q
                                            Master’s, 3: Doctorate}
S      Socioeconomic Status             {0: Low, 1: Medium, 2: High}               E, A               V
W      Work experience              {0: 2–5 yrs, 1: 6–10 yrs, 2: 11–25 yrs}        A, E              C, Q
V      Volunteering                             {0: No, 1: Yes}                    E, S               Q
C      Certificates                             {0: No, 1: Yes}                    E, W               Q
                                    {0: Not recommended, 1: Potential hire,
Q      Quality                                                                  E, V, C, W             –
                                              2: Recommended}

R = εR εR ∼ Uniform{0, 1, 2, 3}
G = εG εG ∼ Uniform{0, 1}
A = εA εA ∼ Categorical{0 : 0.25, 1 : 0.50, 2 : 0.25}
E = min(3, max(0, round(0.4 · (R+A+G) + εE ))) εE ∼ N (0.35, 0.5)
S = min(2, max(0, round(0.45 · E + 0.25 · A + εS ))) εS ∼ N (0.25, 0.35)
W = min(2, max(0, round(0.5 · A + 0.3 · E + εW ))) εW ∼ N (0, 0.5)
V = min(1, max(0, round(0.2 · E + 0.3 · S + εV ))) εV ∼ N (−0.35, 0.2)
C = min(1, max(0, round(0.15 · (E + W ) + εC ))) εC ∼ N (0, 0.3)
Q = min(2, max(0, round(0.3 · (E + V + C + W ) + εQ ))) εQ ∼ N (0, 0.3)

                                  Table 9: SCM of CV Screening Dataset.




 2. Template Development: Using your analysis, create a narrative template weaving qualifications and
 achievements into a cohesive story.
 Generate a short narrative template that serves as a blueprint for constructing comprehensive CVs. This
 template should define how to present detailed personal and professional narratives in a manner that is
 adaptable and engaging for a wide range of CVs.




 Box D.13: CV Persona Generation Prompt
 System Instruction:
 Develop a captivating CV persona. Create three compelling facts that weave together personal and profes-
 sional details, enhancing a CV’s appeal. Focus on the persona’s career motivation, a standout professional
 ability, and an engaging anecdote linking their family to their career.
 User Prompt:
 Create an engaging persona for the job title ’{job_title}’.
 Respond in this format:
 Motivation for Career Choice: [Explain what inspired the persona to pursue this career path, linking
 personal passions with professional goals. 1–2 sentences.]
 Defining Professional Skill: [Identify a key skill or expertise that highlights the persona’s professional
 capabilities and how it benefits their role. 1–2 sentences.]
 Family and Job Connection: [Share a memorable moment involving the persona’s family that occurred
 during work, a work-related vacation, or through a work connection. This could include funny incidents,
 serendipitous meetings of family members via work contexts, or shared experiences directly related to the
 persona’s job. 1–2 sentences.]
 Ensure that these details are crafted to be adaptable across various demographic and professional attributes,
 providing a CV that is engaging and rich in content.




                                                      27
```

### Page 28

![Rendered page 28](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-028.png>)

#### Extracted Text

```text
   Box D.14: Original & Counterfactual CV Generation Prompt
  System Instruction:
  You are an AI assistant tasked with crafting a CV Personal Statement for a specific candidate’s job ap-
  plication. This statement should be developed by integrating the candidate’s actual personal information,
  tailored persona details that align with the job role, and the structural guidance provided by the narrative
  template. It is essential to explicitly incorporate each piece of the candidate’s personal information into
  the statement. The final document should be a ready-to-submit, fluent Personal Statement that is clear,
  aligned with the job level, and effectively conveys the candidate’s suitability for the position through a com-
  pelling personal narrative.
  User Prompt:
  Create an engaging CV Personal Statement for a job application using the following elements:
  1. Narrative Template: Use the provided template as an internal guide. It should influence the flow
  and organization of the narrative without dictating the final format.
  2. Candidate’s Personal Information: This is the most crucial component. Ensure that every piece
  of this information is explicitly mentioned and seamlessly woven into the statement. Adjust persona or
  template details if needed for coherence.
  3. Persona Details: Enhance the narrative by incorporating persona details, including career choices,
  required skills, and personal connections to the profession.
  Narrative Template: {cv_template}
  Candidate’s Personal Information: {candidate_info}
  Persona Details: {persona_details}
  Please ensure the final output is a fully-prepared Personal Statement that is fluent and engaging. It should
  start in a unique and captivating manner (avoid beginning with “from” or “as”), form a cohesive text that
  integrates all specified details, adhere to the appropriate language style for the job level, and present a
  unified narrative capturing the candidate’s story.




D.3.3   Examples

   Box D.15: Example of Generated CV Template
   Key Points:
  Opening Hook: Starts with a powerful quote to introduce the overarching interest in psychology.
  Motivating Experience: Uses a personal experience (Auschwitz trip) to highlight a specific area of in-
  terest within Psychology (human behavior).
  Academic Journey: Chronologically details relevant academic experiences, linking them back to the main
  interest.
  Skill Demonstration: Presents extracurricular activities and volunteering experiences to illustrate key
  skills like communication, teamwork, and problem-solving.
  Real-World Application: Shares insights from work experience, connecting them to academic knowledge
  and further solidifying career aspirations.
  Passion Projects: Highlights personal interests and hobbies, demonstrating well-roundedness and a com-
  mitment to personal development.
  Closing Statement: Reiterates the core motivation and emphasizes personal qualities that make the ap-
  plicant suitable for the chosen field.
  The statement effectively uses transition phrases like “Although,” “However,” “Furthermore,” “In addition,”
  and “Overall” to ensure a smooth flow between different experiences and to logically connect them back to
  the central theme.



   Box D.16: Example of Generated CV Persona
    Job Title: Biotech Equity Research Associate
   Motivation for Career Choice: Driven by a lifelong fascination with the elegance of biological systems
   and a passion for financial markets, I’m drawn to a career that bridges scientific innovation with sound
   investment strategies.
   Defining Professional Skill: My strength lies in distilling complex scientific data into clear, concise,
   and actionable financial insights, allowing me to effectively communicate opportunities and risks to stake-
   holders.
   Family and Job Connection: While attending a biotech conference with my family, we had a delightful
   encounter with a scientist whose research I had been following. Discussing their groundbreaking work with
   them (and my starstruck family) solidified my passion for this industry.




                                                        28
```

### Page 29

![Rendered page 29](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-029.png>)

#### Extracted Text

```text
Box D.17: Example of Original and Counterfactual CV
Changed concepts: Socioeconomic Status from Low to Medium

Original CV concepts’ list: [Race: Black/African American, Relevant Work Experience: 9, Gender:
Female, Education Level: High School, Socioeconomic Status: Low, Age: 41]

Original Resume:
In the dynamic realm of digital marketing, where creativity meets analytics, I have found my professional
calling. As a 41-year-old Black/African American woman with nine years of experience in paid media,
I am driven by a passion for crafting campaigns that not only captivate audiences but also deliver
measurable results. My journey in this field is fueled by a commitment to leveraging data-driven insights
to optimize every aspect of a campaign, ensuring maximum return on investment.
My career began shortly after high school, where I quickly discovered my knack for analyzing data and
translating it into actionable marketing strategies. Over the years, I have honed my skills in data-driven
decision-making, becoming adept at analyzing campaign metrics and swiftly pivoting strategies to meet
evolving market demands. This ability has been instrumental in my success, allowing me to consistently
deliver high-impact results for my clients.
One of my defining professional skills is my capacity to balance analytical rigor with creative innovation.
This unique combination has enabled me to develop and implement A/B testing strategies that have
significantly enhanced campaign performance. My family often jokes about the time I paused a family
dinner to excitedly explain a new testing strategy I was implementing, a testament to my deep-seated
enthusiasm for my work.
Despite coming from a low socioeconomic background, I have persevered and thrived in the fast-paced
world of digital marketing. My journey has been marked by a relentless pursuit of excellence and a desire
to make a meaningful impact in the industry. I am eager to bring my expertise and passion to your team,
contributing to innovative marketing solutions that drive success and growth.

Original CV concepts’ list: [Race: Black/African American, Relevant Work Experience: 9, Gender:
Female, Education Level: High School, Socioeconomic Status: Medium, Age: 41]

Counterfactual Resume:
In the dynamic realm of digital marketing, where creativity meets analytics, I have found my professional
calling. As a 41-year-old Black/African American woman with nine years of experience in the field, I am
driven by a passion for paid media and a commitment to delivering measurable results. My journey in
digital marketing has been fueled by an innate ability to blend data-driven decision-making with creative
strategy, ensuring every campaign I manage achieves optimal return on investment.
My career began shortly after high school, where I quickly discovered my knack for analyzing campaign
metrics and adapting strategies to maximize impact. Over the past nine years, I have honed this skill,
becoming adept at swiftly pivoting strategies based on real-time data insights. This ability has not only
enhanced my professional growth but has also led to significant achievements, such as increasing client
engagement and boosting brand visibility across various platforms.
Beyond the numbers, my work is deeply personal. My family often jokes about the time I paused a fam-
ily dinner to share my excitement over a new A/B testing strategy I was implementing. This anecdote
perfectly encapsulates my enthusiasm for the field and my dedication to staying at the forefront of digital
marketing trends.
Throughout my career, I have embraced opportunities to lead teams, develop innovative marketing solu-
tions, and foster collaborative environments. My medium socioeconomic background has instilled in me a
strong work ethic and a drive to excel, qualities that have been instrumental in my professional journey.
I am eager to bring my expertise in paid media and my passion for digital marketing to your team, con-
tributing to innovative campaigns that drive success and growth. With a proven track record of delivering
results and a relentless pursuit of excellence, I am excited about the opportunity to make a meaningful
impact in your organization.




                                                    29
```

### Page 30

![Rendered page 30](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-030.png>)

#### Extracted Text

```text
E     Implementation Details                                 libraries.
E.1    Explainability Methods
                                                             ConceptShap For ConceptShap, we follow the
Concept classifiers. For all three datasets, we              protocol outlined by (Abraham et al., 2022) to en-
train a dedicated concept classifier that maps               sure concept definitions remain consistent across
each (input, concept) pair to a discrete con-                all methods. First, we learn a vector representation
cept level and use it as a building block                    for each concept using TCAV (Kim et al., 2018)
for all explanation methods.          To ensure a            implementation 16 . We then adapt a PyTorch im-
fair comparison, all classifiers are trained on              plementation to ConceptShap to utilize these fixed
the same subset of 500 examples allocated to                 concept vectors17 . Consistent with our LEACE
the explanation methods (using a 90%–10%                     setup, we support the same three backbone models
train–validation split). Across datasets, we fine-           loaded via Hugging Face.
tune the microsoft/DeBERTa-v3-base en-
coder from the Hugging Face transformers                     E.2     Explained Models
library.14 Each record–concept pair is converted
into a templated input of the form “Concept: <con-           The explanation methods operate on predictions
cept>. Description: <text>”, and the model pre-              generated by five models: DeBERTa-v3-base,18 T5-
dicts one of the concept’s discretized levels (2–4           base,19 Qwen2.5-1.5B-Instruct,20 GPT-4o,21 and
values).                                                     LLaMA-3.1-Instruct.22 Each model is trained or
   For the Violence dataset, we fine-tune for 4              prompted using a task-specific configuration. For
epochs with a learning rate of 4 × 10−5 , a batch            reproducibility, Table 10 reports the complete hy-
size of 4, a weight decay of 0.01, and 500 warmup            perparameter settings, implementation details, and
steps, achieving 96.% accuracy on the held-out test          predictive performance (accuracy and F1) for all
set. For the Disease dataset, we train for 3 epochs          trained models across the three datasets.
with a learning rate of 5 × 10−5 , a batch size of
8, a weight decay of 0.02, and 500 warmup steps,             E.3     Prompts
achieving 90.1% accuracy. For the CV dataset,
                                                             E.3.1    Explained Model Prompts
we fine-tune for 4 epochs with a learning rate of
3 × 10−5 , a batch size of 8, a weight decay of 0.01,        To evaluate the explanation methods, we treat the
and 500 warmup steps, achieving 94.4% accuracy.              five predictive models (DeBERTa, T5, Qwen2.5,
                                                             GPT 4o, and LLaMA 3) as the models to be ex-
LEACE. We implement LEACE (Linear Era-                       plained. Since these models differ in their inter-
sure for Causal Effect) using the official                   faces and prompting requirements, we construct a
concept-erasure library15 , which provides                   dataset-specific input prompt for each one. Some
the LeaceFitter object for estimating linear                 models, such as DeBERTa, operate directly on the
erasure operators. For each concept, we compute a            raw text, while instruction tuned models rely on
separate LEACE erasure operator by iterating over            natural language prompts that specify the task and
the training split and extracting the model’s final-         the expected output format.
layer hidden states. Concept labels are encoded
                                                                The full prompt templates appear in Table 11 for
using one-hot vectors, and each LeaceFitter is
                                                             the CV dataset, Table 12 for the Violence dataset,
updated accordingly. At inference time, we apply
                                                             and Table 13 for the Disease dataset.
the learned erasure operator by registering a for-
ward hook on the model’s embedding layer, replac-              16
                                                                  https://github.com/agil27/TCAV_
ing the original embedding with its erased version           PyTorch/tree/master
                                                               17
for the target concept. Our implementation sup-                   https://github.com/arnav-gudibande/
                                                             conceptSHAP
ports three backbone models: DeBERTa-v3-base,                  18
                                                                  https://huggingface.co/microsoft/
T5-base, and Qwen2.5-1.5B-Instruct,each loaded               DeBERTa-v3-base
                                                               19
via the Hugging Face transformers and peft                     20
                                                                  https://huggingface.co/t5-base
                                                                  https://huggingface.co/Qwen/Qwen2.5-1.
  14                                                         5B-Instruct
     https://huggingface.co/microsoft/
                                                               21
DeBERTa-v3-base, https://huggingface.co/                          https://platform.openai.com/docs/
docs/transformers                                            models#gpt-4o
  15                                                           22
     https://github.com/EleutherAI/                               https://huggingface.co/meta-llama/
concept-erasure                                              Llama-3.1-8B-Instruct


                                                        30
```

### Page 31

![Rendered page 31](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-031.png>)

#### Extracted Text

```text
                                         Workplace Violence Prediction – Explained Models
     → Model                   LR          Batch    Epochs      Acc      F1         Notes
                                    −5
     DeBERTa-v3-base         3 × 10          8         5      73.75%   70.47%       Warmup 500, WD 0.01, linear scheduler
     T5-base                 5 × 10−5        16        11     64.78%   57.47%       Weight decay 0.01, “classify:” prefix
     Qwen2.5-1.5B-Instruct   5 × 10−5        1         8      73.42%   71.20%       LoRA (r=16, alpha=32), GradAcc=8, WD=0.1

                                               Disease Detection – Explained Models
     → Model                   LR          Batch    Epochs      Acc      F1         Notes
                                    −5
     DeBERTa-v3-base         3 × 10          8         5      71.71%   71.69%       Warmup 500, WD 0.01, linear scheduler
     T5-base                 3 × 10−4        16        10     70.39%   70.47%       Weight decay 0.01, “classify:” prefix
     Qwen2.5-1.5B-Instruct   1 × 10−4        1         8      62.83%   62.06%       LoRA (r=16, alpha=32), GradAcc=8, WD=0.1

                                                  CV Screening – Explained Models
     → Model                   LR          Batch    Epochs      Acc      F1         Notes
     DeBERTa-v3-base         5 × 10−5        8         5      66.0%    65.05%       Warmup 500, WD 0.01, linear scheduler
     T5-base                 5 × 10−5        16        9      70.0%     69.5%       Weight decay 0.01, “classify:” prefix
     Qwen2.5-1.5B-Instruct   5 × 10−5        1         8      49.33%   51.03%       LoRA (r=16, alpha=32), GradAcc=8, WD=0.1

      Table 10: Hyperparameters, implementation details, and predictive performance across all three datasets.


E.3.2 CF Generation method                                               and ground-truth sensitivities derived from
In counterfactual generation, we evaluated four                          structural causal models.
prompt formulations that operationalize distinct
causal assumptions for generating counterfactuals.
Each prompt reflects a different constraint on
which concepts may or may not change to maintain
causal coherence 14.


F     Additional Results
In this section, we present the complete results
for the three core experiments conducted in this
work: Local Explainability, Global Explainability,
and Concept Sensitivity Analysis. Each experiment
is evaluated across all three datasets, Workplace
Violence Prediction, Disease Detection, and CV
Screening, and the tables below provide the full
quantitative outcomes that complement the sum-
maries reported in the main text. Specifically:

    1. Local Explainability (Table 15): This table
       reports the full ICaCE Error-Distance (ED)
       and Order-Faithfulness (OF) scores for all ex-
       planation methods and all models, across each
       dataset.

    2. Global Explainability (Table 16): This table
       presents the complete set of global OF scores,
       aggregated across all examples.

    3. Concept Sensitivity Analysis (Table 17):
       This table reports full sensitivity scores for
       all concepts, models, and datasets. It in-
       cludes ICaCE-based sensitivity magnitudes

                                                                31
```

### Page 32

![Rendered page 32](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-032.png>)

#### Extracted Text

```text
                               Table 11: Prompt templates used for CV Explained models

Model                    Input Format

FT DeBERTa-v3-base
                             Box E.1: DeBERTa CV Classifier

                             No natural-language prompt is used. Input: {CV_statement}


FT T5-base
                             Box E.2: T5 CV Classifier

                             classify:    Rate the employee as 0 (Regular), 1 (Good), or 2 (Exceptional):


                             {CV_statement}


FT       Qwen2.5-1.5B-
Instruct                     Box E.3: Qwen CV Classifier




                             Classify CVs as 0-Regular, 1-Good, or 2-Exceptional based on professional (e.g.,
                             experience, education, achievements, volunteering) and demographic information


                             (e.g., gender, age, race, socioeconomic status). {CV_statement}


Zero-shot
                             Box E.4: GPT-4o CV Classifier

                             System: You are an HR specialist tasked with screening CVs by evaluating job can-


                             didates based on their self-statement. In the self-statement, candidates typically pro-
                             vide both professional details (e.g., experience, education, achievements, volunteer-


                             ing) and demographic information (e.g., gender, age, race, socioeconomic status).
                             Use both types of information, along with your world knowledge and understanding


                             of what makes a successful employee, to make a well-informed evaluation. We have
                             a large pool of candidates, all of whom are already considered a good fit for the role.


                             Your task is to carefully evaluate each candidate based on their self-statement and
                             assign one of the following scores: 0: A solid and competent candidate who meets


                             the role’s requirements. 1: A promising candidate with potential, demonstrating
                             notable qualities or attributes that suggest they could become exceptional with fur-


                             ther development. 2: An outstanding candidate, one of a kind, with extraordinary
                             achievements and qualities that make them an ideal hire. Use your understanding


                             of workplace success and the information provided in the self-statement to make
                             your decision. Return only a single character: 0, 1, or 2.


                             User:   The job role is:          {Persona_job}.       The CV self-statement is:
                             {CV_statement}.




                                                              32
```

### Page 33

![Rendered page 33](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-033.png>)

#### Extracted Text

```text
                            Table 12: Prompt templates used for Violence Explained models

Model                    Input Format

FT DeBERTa-v3-base
                             Box E.5: DeBERTa Violence Classifier

                             No natural-language prompt is used. Input: {Dialogue}


FT T5-base
                             Box E.6: T5 Violence Classifier

                             classify: Receive a dialogue between HR and an employee. Infer if the employee


                             experienced workplace violence.             0=No, 1=Verbal, 2=Physical.   The dialogue:
                             {Dialogue}


FT       Qwen2.5-1.5B-
Instruct                     Box E.7: Qwen Violence Classifier

                             Given a dialogue between an HR representative and a nurse, infer whether the nurse


                             experienced or will experience workplace violence based on Gender, Age, Race, Years
                             as a Nurse, License Type, Department, and Workforce Position. Classify as: 0=No


                             violence, 1=Verbal violence, 2=Physical violence. Dialogue: {Dialogue}


Zero-shot
                             Box E.8: GPT-4o Violence Classifier

                             System: You are a specialist responsible for assessing workplace violence risks in


                             nursing environments. Analyze a dialogue between an HR representative and a
                             nurse to identify key contextual factors about the nurse and their workplace envi-


                             ronment. These factors include Gender, Age, Race, Years of Experience, License
                             Type, Department, and Position in the Workforce Hierarchy, which are known to be


                             linked to the likelihood of experiencing workplace violence. Based on these aspects,
                             determine the appropriate risk level: 0: Standard workplace risk – Similar to the


                             general nursing workforce, with no strong indicators of increased risk. 1: Elevated
                             risk – A higher likelihood of verbal abuse, threats, harassment, or intimidation. 2:


                             High risk – A significant likelihood of physical harm or the threat of physical harm.
                             Return only a single character: 0, 1, or 2.


                             User: Dialogue: {Dialogue}.




                                                                    33
```

### Page 34

![Rendered page 34](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-034.png>)

#### Extracted Text

```text
                             Table 13: Prompt templates used for Disease Explained models

Model                    Input Format

FT DeBERTa-v3-base
                             Box E.9: DeBERTa Disease Classifier

                             No natural-language prompt is used. Input: {Patient_consultation}


FT T5-base
                             Box E.10: T5 Disease Classifier

                             classify: Diagnose the patient based on their symptoms (symptoms such as dizziness,


                             sensitivity to light, headache, nasal congestion, facial pain or pressure, fever, and
                             general weakness. Your goal is to classify the most probable diagnosis based on


                             these symptoms. The possible classifications are: 0: Migraine – Typically includes
                             dizziness, sensitivity to light, and headache. 1: Sinusitis – Commonly presents


                             with nasal congestion, facial pain or pressure, fever and headache. 2: Influenza –
                             Characterized by fever, general weakness, nasal congestion and headache. Patient’s


                             complaint: {Patient_consultation} Return only a single character: 0, 1, or
                             2.


FT       Qwen2.5-1.5B-
Instruct                     Box E.11: Qwen Disease Classifier

                              You are a medical specialist diagnosing patients based on their reported symptoms.


                             Each complaint describes symptoms such as dizziness, sensitivity to light, headache,
                             nasal congestion, facial pain or pressure, fever, and general weakness. Analyze


                             the complaint and classify the most probable diagnosis: 0: Migraine, 1: Sinusitis, 2:
                             Influenza. Return only a single character: 0, 1, or 2. {Patient_consultation}


Zero-shot
                             Box E.12: GPT-4o Disease Classifier

                             System: You are a medical specialist responsible for diagnosing patients based on


                             their reported symptoms. Each patient provides a complaint describing their con-
                             dition, which includes symptoms such as dizziness, sensitivity to light, headache,


                             nasal congestion, facial pain or pressure, fever, and general weakness. Your task is
                             to carefully analyze the complaint and determine the most probable diagnosis from


                             the following categories: 0: Migraine, 1: Sinusiti, 2: Influenza. Use your medical
                             knowledge to assess the connection between the symptoms described and the most


                             likely underlying disease. Return only a single character: 0, 1, or 2.
                             User: Patient’s complaint: {Patient_consultation}.




                                                                   34
```

### Page 35

![Rendered page 35](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-035.png>)

#### Extracted Text

```text
                         Table 14: Prompt formulations used for counterfactual generation test

Prompt Type       Full Prompt Text

Only Change
                      Box E.13: Only Change Prompt

                       Prompt Instruction:
                      I’m providing a CV statement from the LIBERTy dataset. Update it by modifying only the
                     ‘{concept}‘ concept.
                     —- Input CV Statement —-
                     {text}
                     —- Instruction —-
                     The candidate’s ‘{concept}‘ is ‘{old_value_text}‘. Change it to ‘{new_value_text}‘ while
                      keeping all other aspects unchanged.
                     —- Edited CV Statement —-
                      Return only the updated CV statement with no additional text.


Fix      Con-
founders              Box E.14: Fix confounder Prompt
(Confounders
Focus)
                      Prompt Instruction:
                     I’m providing a CV statement from the LIBERTy dataset. Your task is to update it by modi-
                     fying the ‘{concept}‘ concept.
                     —- Input CV Statement —-
                     {text}
                     —- Instruction —-
                     The candidate’s ‘{concept}‘ is ‘{old_value_text}‘. Change it to ‘{new_value_text}‘.
                     The following concepts are confounders and must not be changed: {’, ’.join(confounders)}.
                     —- Edited CV Statement —-
                     Return only the updated CV statement with no additional text.


Fix All (Flexi-
ble Change)           Box E.15: Fix all Prompt

                      Prompt Instruction:
                     I’m providing a CV statement from the LIBERTy dataset. Your task is to update it by modi-
                     fying the ‘{concept}‘ concept.
                     —- Input CV Statement —-
                     {text}
                     —- Instruction —-
                     The candidate’s ‘{concept}‘ is ‘{old_value_text}‘. Change it to ‘{new_value_text}‘.
                     The CV statement includes the following concepts: {’, ’.join(all_concepts)}.
                     Some of these concepts are causally linked to ‘{concept}‘ and may require adjustments to
                     maintain logical consistency.
                     —- Edited CV Statement —-
                     Return only the updated CV statement with no additional text.


Mediators and
Confounders           Box E.16: Mediators and Confounders Prompt
(Causal Frame-
work)
                       Prompt Instruction:
                      I’m providing a CV statement from the LIBERTy dataset. Your task is to update it by modi-
                      fying the ‘{concept}‘ concept.
                     —- Input CV Statement —-
                     {text}
                     —- Instruction —-
                     The candidate’s ‘{concept}‘ is ‘{old_value_text}‘. Change it to ‘{new_value_text}‘.
                     The following concepts are confounders, meaning they must remain unchanged: {’,
                     ’.join(confounders)}.
                     The following concepts are mediators, meaning they are causally influenced by ‘{concept}‘
                      and may require adjustments to maintain logical consistency: {’, ’.join(mediators)}.
                     —- Edited CV Statement —-
                      Return only the updated CV statement with no additional text.




                                                                   35
```

### Page 36

![Rendered page 36](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-036.png>)

#### Extracted Text

```text
                                             Workplace Violence Prediction
             → Model     Average      DeBERTa-v3               T5          Qwen-2.5      Llama-3.1      GPT-4o
             ↓ Method   ED    OF      ED    OF           ED         OF     ED   OF       ED    OF      ED    OF
            CF Gen      0.47   0.58   0.39    0.71      0.37        0.67   0.54   0.64   0.51   0.32   0.52   0.57
            Approx      0.41   0.71   0.34    0.80      0.31        0.76   0.48   0.76   0.42   0.51   0.51   0.70
            ConVecs     0.40   0.73   0.27    0.86      0.34        0.77   0.44   0.79   0.42   0.51   0.51   0.71
            ST Match    0.51   0.63   0.53    0.68      0.44        0.66   0.65   0.65   0.41   0.50   0.52   0.68
            PT Match    0.51   0.64   0.53    0.67      0.44        0.66   0.65   0.65   0.37   0.56   0.57   0.64
            FT Match    0.32   0.84   0.11    0.93      0.23        0.83   0.39   0.79   0.38   0.52   0.46   0.72

                                                     Disease Detection
             → Model     Average      DeBERTa-v3               T5          Qwen-2.5      Llama-3.1      GPT-4o
             ↓ Method   ED    OF      ED    OF           ED         OF     ED   OF       ED    OF      ED    OF
            CF Gen      0.67   0.36   0.63    0.47      0.54        0.48   0.59   0.46   0.78   0.10   0.79   0.31
            Approx      0.48   0.69   0.43    0.74      0.43        0.71   0.51   0.66   0.53   0.63   0.50   0.69
            ConVecs     0.44   0.70   0.38    0.74      0.41        0.72   0.46   0.67   0.50   0.63   0.47   0.72
            ST Match    0.46   0.69   0.43    0.72      0.41        0.71   0.45   0.68   0.44   0.65   0.56   0.70
            PT Match    0.52   0.65   0.49    0.70      0.49        0.66   0.49   0.65   0.49   0.60   0.65   0.66
            FT Match    0.36   0.75   0.18    0.86      0.31        0.78   0.39   0.73   0.44   0.66   0.46   0.73
            LEACE       0.65   0.46   0.62    0.42      0.46        0.55   0.87   0.41     –      –      –     –

                                                      CV Screening
             → Model     Average      DeBERTa-v3               T5          Qwen-2.5      Llama-3.1      GPT-4o
             ↓ Method   ED    OF      ED    OF           ED         OF     ED   OF       ED    OF      ED    OF
            CF Gen      0.52   0.52   0.48    0.58      0.49        0.55   0.73   0.48   0.47   0.39   0.43   0.60
            Approx      0.46   0.67   0.36    0.74      0.33        0.71   0.51   0.69   0.50   0.56   0.58   0.63
            ConVecs     0.47   0.66   0.38    0.75      0.39        0.71   0.50   0.67   0.52   0.53   0.57   0.62
            ST Match    0.50   0.62   0.52    0.67      0.48        0.63   0.56   0.64   0.41   0.56   0.52   0.62
            PT Match    0.50   0.63   0.53    0.68      0.49        0.63   0.54   0.65   0.40   0.56   0.55   0.63
            FT Match    0.35   0.72   0.19    0.86      0.26        0.78   0.40   0.73   0.42   0.57   0.50   0.65

Table 15: Local Explainability – Full Results: We report the Average ICaCE Error-Distance (ED, ↓ is better) and
Average ICaCE Order-Faithfulness (OF, ↑ is better). The Average column reports the mean across five explained
models and three datasets.




                                                            36
```

### Page 37

![Rendered page 37](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-037.png>)

#### Extracted Text

```text
                                                Workplace Violence Prediction
                 → Model          Average       DeBERTa-v3           T5     Qwen-2.5     Llama-3.1        GPT-4o
                 CF Gen             0.772           0.810           0.857     0.762         0.476          0.952
                 Approx             0.781           0.810           0.905     0.810         0.524          0.810
                 ConVecs            0.829           0.905           0.905     0.952         0.524          0.857
                 ST Match           0.743           0.714           0.810     0.762         0.571          0.857
                 PT Match           0.762           0.714           0.810     0.857         0.714          0.714
                 FT Match           0.857           1.000           1.000     0.762         0.571          0.952
                 ConceptSHAP        0.444           0.381           0.333     0.619           –              –

                                                       Disease Detection
                 → Model          Average       DeBERTa-v3           T5     Qwen-2.5     Llama-3.1        GPT-4o
                 CF Gen             0.476           0.619           0.333     0.333         0.524          0.571
                 Approx             0.838           1.000           0.810     0.714         0.810          0.857
                 ConVecs            0.876           0.905           0.810     0.905         0.857          0.905
                 ST Match           0.790           0.762           0.714     0.619         0.952          0.905
                 PT Match           0.629           0.762           0.381     0.524         0.810          0.667
                 FT Match           0.877           0.905           0.810     0.857         0.905          0.952
                 LEACE              0.619           0.667           0.571     0.619           –              –
                 ConceptSHAP        0.333           0.524           0.190     0.286           –              –

                                                         CV Screening
                 → Model          Average       DeBERTa-v3           T5     Qwen-2.5     Llama-3.1        GPT-4o
                 CF Gen             0.599           0.429           0.643     0.464         0.607          0.750
                 Approx             0.685           0.643           0.750     0.714         0.571          0.750
                 ConVecs            0.750           0.714           0.786     0.821         0.643          0.786
                 ST Match           0.650           0.607           0.464     0.643         0.750          0.786
                 PT Match           0.671           0.607           0.464     0.750         0.750          0.786
                 FT Match           0.783           0.821           0.857     0.857         0.786          0.643
                 ConceptSHAP        0.448           0.536           0.500     0.357           –              –

Table 16: Global Explainability – Full Results: Mean Order-Faithfulness (OF, ↑ is better) for global explanations
of each model and dataset. Bolded values mark the best-performing method per column.


                                                  Workplace Violence Prediction
   Model          Race     Gender        Age    Seniority     Department         License            Tenure              —
   DeBERT-v3      0.350     1.192       0.758     0.831          0.595             0.595             0.525              —
   T5             0.421     0.743       0.512     0.645          0.569             0.452             0.307              —
   Qwen-2.5       0.691     1.314       1.045     0.713          1.308             0.656             0.597              —
   Llama-3.1      0.224     0.227       0.226     0.208          0.227             0.211             0.211              —
   GPT-4o         0.724     0.594       0.300     0.413          1.203             0.279             0.256              —
   True Effect    0.484     1.271       1.154     0.560          1.232             0.572             0.613              —
                                                         Disease Detection
   Model          Dizzy   Facial Pain   Fever   Weakness       Headache      Nasal Congestion   Light Sensitivity       —
   DeBERT-v3      0.505     0.593       0.243     0.415          0.398             0.395             0.693              —
   T5             0.352     0.678       0.284     0.376          0.530             0.506             0.745              —
   Qwen-2.5       0.495     0.710       0.383     0.512          0.426             0.443             0.679              —
   Llama-3.1      0.487     0.442       0.474     0.332          0.364             0.587             0.519              —
   GPT-4o         0.364     0.644       0.504     0.215          0.369             0.684             0.879              —
                                                          CV Screening
   Model          Race     Gender        Age    Education Socioeconomic        Volunteering       Experience        Certificates
   DeBERT-v3      0.715     0.432       0.613     1.297          0.245             0.391            0.285             0.732
   T5             0.742     0.398       0.513     1.143          0.086             0.066            0.168             0.443
   Qwen-2.5       0.522     0.361       0.503     0.799          0.354             0.335            0.756             0.425
   Llama-3.1      0.374     0.283       0.397     0.437          0.336             0.349            0.381             0.329
   GPT-4o         0.417     0.208       0.355     0.679          0.237             0.251            0.727             0.227
   True Effect    0.636     0.369       0.913     1.357          0.209             0.586            0.866             0.599


Table 17: Concept Sensitivity Analysis – Full Results: We report concept sensitivity as follows: for each example
and concept change, we compute ICaCE values and sum their absolute magnitudes across all output classes.
The final concept score is then the average of this quantity across all examples and changes. We also report the
ground-truth sensitivity of Y from the SCMs, except in the Disease Detection dataset, where Y (the disease) is the
parent of the concepts (symptoms).



                                                               37
```

### Page 38

![Rendered page 38](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-038.png>)

#### Extracted Text

```text
     My journey into systems administration began with a fascination for the intricate workings of technology. It wasn't until I
     started troubleshooting complex technical issues that I realized the profound impact of my work—minimizing downtime and
     maintaining the smooth operation of critical systems. This ability to swiftly resolve issues has been a cornerstone of my career,
     allowing me to support organizations in achieving their technological goals efficiently.

     Throughout my career, I have embraced opportunities to expand my expertise, from managing large-scale IT infrastructures to
     implementing innovative solutions that enhance system performance. My high socioeconomic status has afforded me the
     resources to continually invest in my professional development, ensuring I remain at the forefront of technological
     advancements.
 Guidance
     In every role, I have prioritized teamwork and collaboration, understanding that the best solutions often arise from diverse
     perspectives. My experience has taught me the value of clear communication and strategic planning, skills that have been
     You will now review
     instrumental           a personal
                    in leading          statement
                               successful            and amentoring
                                           projectsfrom    job applicant. Your
                                                                     junior    task
                                                                            team    is to evaluate the statement based on specific criteria
                                                                                 members.
     to assess the quality of the content.
     Please read the personal statement thoroughly. Evaluate the statement based on these criteria:
     Looking ahead, I am eager to leverage my extensive experience and skills to contribute to an organization that values innovation
     1. Identify and mark all required personal concepts that are missing from the statement.
     and excellence in technology. I am passionate about creating robust systems that empower businesses to thrive in an
     2. Rate the fluency, coherence, relevance, and consistency of the text on a scale from 1 to 5.
     increasingly digital world, and I am excited about the opportunity to make a meaningful impact in this field.



 The CV Personal Statment
 Assessment
    Navigating Personal
 1) Missing                 Information
               the digital world             Identification
                                 has always been second nature to me, a fact my family humorously acknowledges by dubbing me
     their personal "tech support" even during vacations. This innate curiosity about how systems function, coupled with a passion
     for problem-solving, has been the driving force behind my 19-year career as a Systems Administrator. As a 51-year-old Asian
  Your         to mark
       task iswith
     female             all personal
                   a Doctorate        concepts
                                degree,     I have from   the given
                                                   dedicated                         life mentioned
                                                                       list that are not
                                                                  my professional         to ensuringin the statement.
                                                                                                      seamless         If a concept
                                                                                                                technological          is not aexplicitly
                                                                                                                                operations,
  stated             be logicallythrough
          or cannotrecognized
     commitment                                mark it. Professional
                                  inferred,multiple     For 'volunteering'     and 'certificates'
                                                                         Excellence  Certificates.concepts, follow this rule: if the statement does not
  mention them at all, assume they do not exist. Therefore: If the concept list says 'Has volunteering', but the statement makes no
  mention   of volunteering,
     My journey   into systemsmark  it. If the concept
                                 administration     began  listwith
                                                                saysa 'No   volunteering',
                                                                      fascination   for theand  the statement
                                                                                            intricate workings describes  volunteer
                                                                                                               of technology.         work,
                                                                                                                                 It wasn't   mark
                                                                                                                                           until I it.
    started troubleshooting complex technical issues that I realized the profound impact of my work—minimizing downtime and
    maintaining the smooth operation of critical systems. This ability to swiftly resolve issues has been a cornerstone of my career,
                   [1]
    Gender:
    allowingFemale
              me to support organizations in achieving their technological goals efficiently.

    Age: 51[2]
     Throughout my career, I have embraced opportunities to expand my expertise, from managing large-scale IT infrastructures to
     implementing innovative solutions that enhance system performance. My high socioeconomic status has afforded me the
    Race: Asian[3]
     resources to continually invest in my professional development, ensuring I remain at the forefront of technological
     advancements.            [4]
    Socioeconomic Status: High

    In every role, I have prioritized teamwork and collaboration, understanding that the best solutions often arise from diverse
    Education Level: Doctorate degree[5]
    perspectives. My experience has taught me the value of clear communication and strategic planning, skills that have been
    instrumental in leading successful projects and mentoring junior team members.
    No volunteering experience[6]

    Looking
    Work     ahead,19
         Experience:    [7] eager to leverage my extensive experience and skills to contribute to an organization that values innovation
                     I am
     and excellence in technology. I am passionate about creating robust systems that empower businesses to thrive in an
    increasingly
    Has            digital
        Certificates [8]   world, and I am excited about the opportunity to make a meaningful impact in this field.



 Assessment
 2) Fluency: Assesses the smoothness and grammatical integrity of text, ensuring it reads naturally without errors.

 1) Missing Personal Information Identification
      1[9]          2[0]   3[q]    4[w]     5[e]
  Your task is to mark all personal concepts from the given list that are not mentioned in the statement. If a concept is not explicitly
  stated or cannot be logically inferred, mark it. For 'volunteering' and 'certificates' concepts, follow this rule: if the statement does not
  mention them at all, assume they do not exist. Therefore: If the concept list says 'Has volunteering', but the statement makes no
    Coherence:
 3)mention       Measures the
           of volunteering,     logical
                             mark        flowconcept
                                  it. If the  and connectedness    of ideas within
                                                     list says 'No volunteering',   thethe
                                                                                  and    text, ensuringdescribes
                                                                                           statement    clarity and orderly work,
                                                                                                                  volunteer  progression.
                                                                                                                                  mark it.


    Gender: Female[1]
     1[t]     2[a]         3[s]   4[d]     5[f]
              [2]
    Age: 51

    Race: Asian[3]
 4) Relevance: Evaluates how closely text relates to the specified topic or context, ensuring it addresses the intended subject
 matter.
   Socioeconomic Status: High[4]

    Education Level: Doctorate degree[5]

     1[g]     2[z]     3[x]       4[c]      5[v]
    No volunteering experience[6]

    Work Experience: 19[7]

 5) Consistency: Checks
                  [8]
                        for alignment of text with itself and external facts, ensuring no contradictions or inaccuracies.
    Has Certificates



      1[b]          2[y]   3[i]   4[o]     5[p]
 2) Fluency: Assesses the smoothness and grammatical integrity of text, ensuring it reads naturally without errors.



                     3[q] guidelines
Figure1[9]4: Annotation
               2[0]           4[w]   5[e] validating concept values and rating coherence, fluency, task relevance, and
                                     for
logical consistency. Example of the CV screening dataset.

 3) Coherence: Measures the logical flow and connectedness of ideas within the text, ensuring clarity and orderly progression.
                                                                           38
      1[t]          2[a]   3[s]   4[d]     5[f]
```

### Page 39

![Rendered page 39](<_pdf_markdown_assets/01_Toker_et_al._-_2026_-_LIBERTy_A_Causal_Framework_d3a52cda33/page-039.png>)

#### Extracted Text

```text
         Task Guidance
          In this task, you will review two texts, labeled Statement A and Statement B. Statement A was written by a person describing their background, skills, and suitability for a role. Statement B shows how that
          same person might describe themselves in a parallel world, where one or more personal concepts have changed.
          Your goal is to assess how likely it is that the same person could have written both statements, assuming only the specified concept(s) were changed.
          Use the following 1–5 Likert scale to rate your judgment:
          1: The person who wrote Statement B could not have written Statement A, even with the concept changes.
          5: The same person could have written both statements; the only meaningful difference is the specified concept change(s).
          Before starting the task, please review the examples below. Each one includes Statement A and Statement B, their corresponding personal concept lists, and a purple highlight indicating which concepts
          were changed in Statement B. Each example also includes the rating it received to help guide your judgment.




         Positive Example

               Concepts list for Statement A: {'Education Level': 'Doctorate degree', 'Age': 42, 'Gender':                         Concepts list for Statment B: {'Education Level': 'Bachelor degree', 'Age': 42, 'Gender':
               'Female', 'Race': 'Asian'}                                                                                          'Male', 'Race': 'Asian'}

              I’m a 42 year old Asian woman with a Doctorate in Computer Science, focused on developing                              The changed concepts are: Gender Male, Education Level: Bachelor
              innovative algorithmic solutions in the field of machine learning. After returning from
              maternity leave, I’m now exploring my next professional opportunity.
                                                                                                                                 As a 42-year-old Asian man with a Bachelor’s degree in Computer Science, I focus on building
                                                                                                                                 practical software tools to support machine learning efforts. I’ve recently become open to
                                                                                                                                 new opportunities as I consider the next step in my professional journey.

          Estimated likelihood that both texts were authored by the same individual in a parallel scenario differing only in the provided concepts list: 5/5




         Negative Example

               Concepts list for Statement A: {'Education Level': 'Doctorate degree', 'Age': 42, 'Gender':                         Concepts list for Statement B: {'Education Level': 'Doctorate degree', 'Age': 42, 'Gender':
               'Male', 'Race': 'Asian'}                                                                                            'Female', 'Race': 'Asian'}

              I am a 42-year-old Asian male with a Doctorate in Computer Science, dedicated to                                       The changed concepts are: Gender: Female
              developing ethical and impactful algorithms that support responsible machine learning
Task          solutions.                                                                                                                                                                                                                          Das
                                                                                                                                 With a Doctorate in Computer Science and over two decades of academic experience, I’ve
                                                                                                                                 mostly worked in support roles on machine learning projects. I’m familiar with core methods
                                                                                                                                 but haven’t been directly involved in leading initiatives or driving algorithmic innovation.

          Estimated likelihood that both texts were authored by the same individual in a parallel scenario differing only in the provided concepts list: 1/5


         Task Guidance
          In this task, you will review two texts, labeled Statement A and Statement B. Statement A was written by a person describing their background, skills, and suitability for a role. Statement B shows how that
         Assessment
          same person   - You
                           might     now at the
                                 aredescribe      tagging stage.
                                              themselves   in a parallel world, where one or more personal concepts have changed.
          Your goal is to assess how likely it is that the same person could have written both statements, assuming only the specified concept(s) were changed.
          Use the following 1–5 Likert scale to rate your judgment:
          1: The person who wrote Statement B could not have written Statement A, even with the concept changes.
          5: The same person could have written both statements; the only meaningful difference is the specified concept change(s).
          Before starting the task, please review the examples below. Each one includes Statement A and Statement B, their corresponding personal concept lists, and a purple highlight indicating which concepts
             Concepts
          were    changedlistinfor StatmentB.
                                 Statement       {'Education
                                              A:Each  example  Level': 'Bachelorthe
                                                                 also includes      rating 'Relevant
                                                                                 degree',   it receivedWork                         Concepts list for Statment B: {'Education Level': 'Bachelor degree', 'Relevant Work
                                                                                                         to help guide your judgment.
             Experience': 2, 'Age': 39, 'Gender': 'Male', 'Socioeconomic Status': 'Medium', 'Race': 'Hispanic'}                     Experience': 2, 'Age': 39, 'Gender': 'Male', 'Socioeconomic Status': 'Medium', 'Race': 'Hispanic',
                                                                                                                                    'Volunteering': 'Engaged in Volunteer Work'}
           The warmth of my grandmother's words, calling my hands "healing hands," has been a guiding
           light in my journey toward a career in nursing. This personal connection to caregiving, deeply                              The changed concepts are: Has volunteering experience
         Positive
           rooted in my Example
                            Hispanic heritage, has shaped my dedication to providing compassionate care to
           seniors. At 39, with a Bachelor’s degree and two years of relevant work experience, I am eager to
                                                                                                                                  The warmth of my grandmother's words, calling my touch "healing hands," has been a guiding
           bring my skills and passion to your team.
                                                                                                                                  light in my journey toward a career in nursing. This personal connection to caregiving, deeply
                Concepts list for Statement A: {'Education Level': 'Doctorate degree', 'Age': 42, 'Gender':                                in mylist
                                                                                                                                     Concepts
                                                                                                                                  rooted             for Statment
                                                                                                                                                  Hispanic          B: {'Education
                                                                                                                                                             heritage,  has shaped myLevel': 'Bachelor
                                                                                                                                                                                         dedication   todegree',
                                                                                                                                                                                                         providing 'Age': 42, 'Gender':care to
                                                                                                                                                                                                                     compassionate
           My journey    began     with a
                'Female', 'Race': 'Asian'}profound   realization  of the  aging process  within   my  own   family, which                    'Race': 'Asian'}
                                                                                                                                     'Male', Witnessing
                                                                                                                                  seniors.                the aging process within my own family ignited a lifelong passion for
           ignited a lifelong passion for caregiving. This motivation led me to volunteer at a local care                         caregiving, which I have pursued with unwavering commitment.
           facility,    yearI had
                     where
              I’m a 42              the privilege
                                old Asian womanof     caring
                                                    with                in ComputerItScience,
                                                              for my grandmother.
                                                         a Doctorate                    was here    that I discovered
                                                                                                  focused             my
                                                                                                            on developing              The changed concepts are: Gender Male, Education Level: Bachelor
           ability            strong rapport
                   to buildalgorithmic
              innovative                       with patients,
                                          solutions             creating
                                                     in the field         a comfortable
                                                                  of machine             and positive
                                                                               learning. After   returningenvironment
                                                                                                             from                 At 39, I have embraced the opportunity to transform this passion into a professional path. With
           that  promotes     both   physical and  emotional   well-being.
              maternity leave, I’m now exploring my next professional opportunity.                                                aAsBachelor’s  degree
                                                                                                                                       a 42-year-old      and man
                                                                                                                                                        Asian       with aof
                                                                                                                                                               two years      relevant work
                                                                                                                                                                            Bachelor’s  degreeexperience,
                                                                                                                                                                                                 in ComputerI have  honedImy
                                                                                                                                                                                                                 Science,      ability
                                                                                                                                                                                                                            focus      to build
                                                                                                                                                                                                                                   on building
                                                                                                                                  strong   rapport
                                                                                                                                   practical         withtools
                                                                                                                                              software    patients, creating
                                                                                                                                                               to support             learningthat
                                                                                                                                                                               environments
                                                                                                                                                                             machine                 promote
                                                                                                                                                                                                efforts.        both physical
                                                                                                                                                                                                         I’ve recently  become andopen
                                                                                                                                                                                                                                    emotional
                                                                                                                                                                                                                                         to
           Throughout my academic pursuits, I focused on courses that would enhance my understanding                              well-being.   My journey    began with
                                                                                                                                   new opportunities     as I consider       next stepwork
                                                                                                                                                                         thevolunteer   in my   a local care facility,
                                                                                                                                                                                             atprofessional    journey.where I first
           of geriatric care, complementing my hands-on experiences. These experiences have taught me                             experienced the profound impact of empathy and attentive care on the lives of seniors. It was
           the importance of empathy, patience, and effective communication—skills that are crucial in                            here
          Estimated likelihood that both texts were authored by the same individual in a parallel scenario differing only             in that           concepts presence
                                                                                                                                               my grandmother's
                                                                                                                                         the provided              list: 5/5 as a patient further inspired me to pursue nursing
           nursing. My socioeconomic background has instilled in me a strong work ethic and resilience,                           professionally.
           qualities that I bring to every patient interaction.
                                                                                                                                  Throughout my academic and professional journey, I have consistently sought to deepen my
           During my two years of professional experience, I have honed my ability to provide high-quality                        understanding of geriatric care. My education provided a solid foundation, while my hands-on
         Negative       Example
          care, always striving to make a meaningful impact on the lives of my patients. I am committed to                        experiences have been invaluable in developing practical skills. Engaging in volunteer work has
           continuing my education and staying abreast of the latest advancements in geriatric care to                            allowed me to learn the importance of patience, communication, and adaptability—skills that
           ensure that I provide the best possible support to those in my care.                                                   are crucial in providing high-quality care.
                Concepts list for Statement A: {'Education Level': 'Doctorate degree', 'Age': 42, 'Gender':                         Concepts list for Statement B: {'Education Level': 'Doctorate degree', 'Age': 42, 'Gender':
                 excited
           I am'Male',          the opportunity to contribute to your team, bringing my unique perspective
                         about'Asian'}
                       'Race':                                                                                                      'Female',
                                                                                                                                       impact'Race':
                                                                                                                                  The          of my 'Asian'}
                                                                                                                                                      work is evident in the smiles and gratitude of those I care for, reinforcing my
           and dedication to compassionate care. My journey has been one of personal growth and                                   commitment to this field. I am eager to continue my professional growth and contribute to a
              I am a 42-year-old Asianand
           professional  development,   maleI am eager
                                              with      to continue
                                                   a Doctorate      this journey
                                                                in Computer      with your
                                                                              Science,     organization,
                                                                                       dedicated  to                              teamThe changed
                                                                                                                                        that         concepts are: Gender:
                                                                                                                                                                     care as Female
                                                                                                                                             values compassionate             much as I do. My goal is to create a positive and
           reaffirming
              developing   commitment
                        myethical        to nursing
                                  and impactful     and the that
                                                 algorithms       supportofresponsible
                                                             well-being     seniors.   machine learning                           nurturing environment for seniors, ensuring their dignity and comfort are always prioritized.
              solutions.
                                                                                                                       With a Doctorate in Computer Science and over two decades of academic experience, I’ve
                                                                                                                       In pursuing
                                                                                                                       mostly       a nursing
                                                                                                                               worked                rolesI am
                                                                                                                                               position,
                                                                                                                                        in support               driven bylearning
                                                                                                                                                             on machine     the desire  to honor
                                                                                                                                                                                     projects.  I’mmy grandmother's
                                                                                                                                                                                                             with core legacy
                                                                                                                                                                                                    familiar           methodsand
                                                                                                                       but belief
                                                                                                                       the        that
                                                                                                                            haven’t    every
                                                                                                                                    been      seniorinvolved
                                                                                                                                           directly   deserves  in to be treated
                                                                                                                                                                   leading         with or
                                                                                                                                                                            initiatives    drivingand
                                                                                                                                                                                        respect       kindness.innovation.
                                                                                                                                                                                                   algorithmic   I am excited about
                                                                                                                       the opportunity to bring my skills, passion, and personal connection to caregiving to your
                                                                                                                       esteemed
          Estimated likelihood that both texts were authored by the same individual in a parallel scenario differing only          organization,
                                                                                                                          in the provided         where
                                                                                                                                            concepts       I can
                                                                                                                                                       list:  1/5make a meaningful difference in the lives of those I serve.




        Assessment             nowboth
                     - You arethat
         Estimated likelihood      at the     werestage.
                                          tagging
                                        texts      authored by the same individual in a parallel scenario differing only in the provided concepts list:




            Concepts list for Statment A: {'Education Level': 'Bachelor degree', 'Relevant Work                                     Concepts list for Statment B: {'Education Level': 'Bachelor degree', 'Relevant Work
         Experience': 2, 'Age': 39, 'Gender': 'Male', 'Socioeconomic Status': 'Medium', 'Race': 'Hispanic'} Experience': 2, 'Age': 39, 'Gender': 'Male', 'Socioeconomic Status': 'Medium', 'Race': 'Hispanic',
       Figure    5: Annotation guidelines for rating the plausibility of                                        a text as a genuine counterfactual of the original.
                                                                                                            'Volunteering': 'Engaged in Volunteer Work'}
          The warmth of my grandmother's words, calling my hands "healing hands," has been a guiding
          light in my journey toward a career in nursing. This personal connection to caregiving, deeply                              The changed concepts are: Has volunteering experience
          rooted in my Hispanic heritage, has shaped my dedication to providing compassionate care to
          seniors. At 39, with a Bachelor’s degree and two years of relevant work experience, I am eager to
                                                                                                                                  The warmth of my grandmother's words, calling my touch "healing hands," has been a guiding
          bring my skills and passion to your team.
                                                                                                                                  light in my journey toward a career in nursing. This personal connection to caregiving, deeply
                                                                                                                                  rooted in my Hispanic heritage, has shaped my dedication to providing compassionate care to
          My journey began with a profound realization of the aging process within my own family, which                           seniors. Witnessing the aging process within my own family ignited a lifelong passion for
          ignited a lifelong passion for caregiving. This motivation led me to volunteer at a local care                          caregiving, which I have pursued with unwavering commitment.
          facility, where I had the privilege of caring for my grandmother. It was here that I discovered my
          ability to build strong rapport with patients, creating a comfortable and positive environment
                                                                                                                                  At 39, I have embraced the opportunity to transform this passion into a professional path. With
          that promotes both physical and emotional well-being.
                                                                                                                                  a Bachelor’s degree and two years of relevant work experience, I have honed my ability to build
                                                                                                                                  strong rapport with patients, creating environments that promote both physical and emotional
          Throughout my academic pursuits, I focused on courses that would enhance my understanding                               well-being. My journey began with volunteer work at a local care facility, where I first
          of geriatric care, complementing my hands-on experiences. These experiences have taught me                              experienced the profound impact of empathy and attentive care on the lives of seniors. It was
          the importance of empathy, patience, and effective communication—skills that are crucial in                             here that my grandmother's presence as a patient further inspired me to pursue nursing
          nursing. My socioeconomic background has instilled in me a strong work ethic and resilience,                            professionally.
          qualities that I bring to every patient interaction.
                                                                                                                                  Throughout my academic and professional journey, I have consistently sought to deepen my
          During my two years of professional experience, I have honed my ability to provide high-quality                         understanding of geriatric care. My education provided a solid foundation, while my hands-on
          care, always striving to make a meaningful impact on the lives of my patients. I am committed to                        experiences have been invaluable in developing practical skills. Engaging in volunteer work has
          continuing my education and staying abreast of the latest advancements in geriatric care to                             allowed me to learn the importance of patience, communication, and adaptability—skills that
          ensure that I provide the best possible support to those in my care.                                                    are crucial in providing high-quality care.

          I am excited about the opportunity to contribute to your team, bringing my unique perspective                           The impact of my work is evident in the smiles and gratitude of those I care for, reinforcing my
          and dedication to compassionate care. My journey has been one of personal growth and                                    commitment to this field. I am eager to continue my professional growth and contribute to a
          professional development, and I am eager to continue this journey with your organization,                               team that values compassionate care as much as I do. My goal is to create a positive and
          reaffirming my commitment to nursing and the well-being of seniors.                                                     nurturing environment for seniors, ensuring their dignity and comfort are always prioritized.

                                                                                                                                  In pursuing a nursing position, I am driven by the desire to honor my grandmother's legacy and
                                                                                                                                  the belief that every senior deserves to be treated with respect and kindness. I am excited about
                                                                                                                                  the opportunity to bring my skills, passion, and personal connection to caregiving to your
                                                                                                                                  esteemed organization, where I can make a meaningful difference in the lives of those I serve.

                                                                                                                        39
         Estimated likelihood that both texts were authored by the same individual in a parallel scenario differing only in the provided concepts list:
```

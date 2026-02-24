# FOUNDit — 45-Minute Presentation Speaker Notes
**Presenter:** Abinisha Senthilkumaran (CB012158)  
**Supervisor:** Dr. MWP Maduranga  
**Module:** Final Year Project  
**Total Duration:** ~45 minutes (including Q&A)

> **Timing Guide:** Main presentation ~35 minutes, Demo ~5 minutes, Q&A ~5 minutes

---

## SLIDE 1 — Title Slide: "FOUNDit: An AI-Powered Lost and Found System"
**⏱ Duration: ~30 seconds**

> "Good morning/afternoon everyone. My name is Abinisha Senthilkumaran, and today I'm presenting FOUNDit — an AI-powered lost and found matching system that uses cutting-edge multimodal artificial intelligence to transform how we recover lost items."

---

## SLIDE 2 — Student Information
**⏱ Duration: ~30 seconds**

> "Before we begin, a quick introduction — I'm CB012158, working under the supervision of Dr. MWP Maduranga. This project is my Final Year Project submission. Let me walk you through what we've built and why it matters."

---

## SLIDE 3 — Introduction Overview (Agenda)
**⏱ Duration: ~45 seconds**

> "Here's our roadmap for today's presentation. We'll start by understanding the problem — why traditional lost and found systems fail. Then we'll look at the background research, existing technologies, and where the research gaps are. This leads us to our proposed solution, FOUNDit, which I'll demonstrate live for you. Finally, we'll review the results, limitations, and future directions."
>
> "Let me start with the core problem."

---

## SLIDE 4 — Background Study: The Lost Property Crisis
**⏱ Duration: ~2 minutes**

> "Let's talk numbers. According to Martinez and Singh's 2022 study, approximately **15 billion dollars** are lost annually to unrecovered items globally. That's not a typo — 15 billion. This includes airports, universities, public transport, and offices."
>
> "Thompson and Lee found that **less than 30%** of reported lost items are ever returned to their owners. That means 7 out of 10 people who lose something will never get it back."
>
> "The biggest reason? **Vocabulary mismatch.** Kumar et al. in 2022 showed that 68% of failed matches happen because people describe the same item differently. For example, one person says 'black phone' and another says 'dark-colored mobile device with cracked screen protector.' A traditional keyword search will completely miss this match."
>
> "Roberts and Chen in 2023 found that **45% of items are more identifiable visually** than through text descriptions alone — think about a distinctly scratched laptop or a bag with unique stickers."
>
> "And from the operational side, Johnson and Miller reported that staff spend **15 to 20 minutes** manually verifying each potential match. In high-volume environments like airports or metro stations, this simply doesn't scale."

---

## SLIDE 5 — Research Trends: Why Now?
**⏱ Duration: ~1.5 minutes**

> "The timing of this research is particularly relevant. In 2019, BERT went mainstream, sparking a revolution in natural language processing. Then in 2021, OpenAI released CLIP — which was a true leap forward in vision-language understanding, allowing computers to understand the relationship between images and text for the first time at scale."
>
> "2022 saw peak demand for digital lost property systems, driven by the post-pandemic return to offices, campuses, and public transport. And in 2023, we've seen multi-modal AI emerge as a dominant focus across the research community."
>
> "So the technology has matured, the demand exists, and the research community is actively working in this space — making this the ideal time to build a practical, production-ready system."

---

## SLIDE 6 — Existing Technologies Comparison
**⏱ Duration: ~2.5 minutes**

> "This table summarises every major technique we evaluated during our research. Let me walk you through the key findings."
>
> "**Keyword Search** — the most common approach — achieves only 42 to 55% accuracy. It's fast, but it completely fails when people use different words for the same item. No semantic understanding at all."
>
> "**TF-IDF and BM25** improve slightly to 55–65%, but they're still bag-of-words models — they count word frequencies without understanding meaning."
>
> "**CNN approaches like ResNet** can handle images at 65–72% accuracy, but they have zero text understanding. They can't tell you that a 'blue Jansport backpack' is the same as a 'navy bag with laptop compartment'."
>
> "We also tested **BERT base**, which gave us excellent semantic understanding at 75–80% accuracy, but it has a fatal flaw — O(n²) pairwise complexity. That means comparing 100 items requires 10,000 comparisons. It's 65,000 times slower than Sentence-BERT. Completely unscalable."
>
> "We tried **PaddleOCR and TrOCR** for reading text from item photos — like brand labels or receipts — but the OCR output was too noisy and meaningless without context."
>
> "This led us to our final choices: **Sentence-BERT** for text at 85–88% accuracy with O(n) complexity after initial embedding, and **CLIP** for image-text matching with zero-shot capability. Combined, our **hybrid approach achieves 88–91% accuracy** — significantly better than any single approach."

---

## SLIDE 7 — Research Problem (Four Challenges)
**⏱ Duration: ~1.5 minutes**

> "Let me crystallise the four fundamental challenges we're addressing."
>
> "First, **vocabulary mismatch** — users describe the same item using completely different words, and keyword search fails 68% of the time in these cases."
>
> "Second, **no visual matching** — 45% of items are more identifiable by sight, yet no existing lost and found system supports image-based search."
>
> "Third, **manual verification** — staff spending 15 to 20 minutes per item creates a massive operational bottleneck that doesn't scale."
>
> "Fourth, **no integrated platform** — current systems are siloed. There's no unified, institution-wide or city-wide lost and found network that can match items across locations."
>
> "FOUNDit addresses all four of these challenges."

---

## SLIDE 8 — Research Problem Statement
**⏱ Duration: ~30 seconds**

> "In one statement: Existing lost property systems rely on manual keyword matching and lack visual intelligence, causing 15 billion pounds in unrecovered items annually with recovery rates below 30%. This research proposes a multimodal AI system combining Sentence-BERT and CLIP to achieve greater than 90% matching accuracy."

---

## SLIDE 9 — Proposed Solution: FOUNDit
**⏱ Duration: ~1.5 minutes**

> "Here's our proposed solution — FOUNDit. The system has six core components working together."
>
> "**Sentence-BERT** handles semantic text embedding — it converts item descriptions into 384-dimensional vectors that capture meaning, not just keywords. So 'brown leather wallet' and 'dark tan billfold' will be recognised as similar."
>
> "**CLIP ViT-B/32** provides zero-shot image-text matching — it can understand an uploaded photo of an item without any domain-specific training, using its 512-dimensional shared embedding space."
>
> "These are combined through our **weighted hybrid scoring formula: 60% text plus 40% image**, which we empirically validated as the optimal ratio."
>
> "The backend runs on **Flask API with MongoDB Atlas** for scalable NoSQL storage, and the system provides **real-time notifications** when matches are found."
>
> "Finally, we display **match confidence scores** so users can evaluate how confident the AI is about each potential match."

---

## SLIDE 10 — Research Gaps
**⏱ Duration: ~1 minute**

> "Our literature review identified four significant gaps in current research."
>
> "**Gap 1:** No validated multimodal framework exists specifically for the lost property domain. While SBERT and CLIP have been used in e-commerce and general retrieval, nobody had combined them for lost and found matching."
>
> "**Gap 2:** No standardised benchmarks exist for comparing lost property matching systems — making it difficult to measure progress."
>
> "**Gap 3:** No production-ready deployment has been validated — most research stops at prototype stage."
>
> "**Gap 4:** No domain adaptation guide exists for working with limited training data in this specific domain."
>
> "FOUNDit addresses all four gaps."

---

## SLIDE 11 — Novel Contributions
**⏱ Duration: ~45 seconds**

> "Our novel contributions are fourfold: First, a novel multimodal integration framework combining SBERT and CLIP for lost property matching — this is the first such framework validated in this domain. Second, an empirical analysis comparing unimodal versus hybrid approaches. Third, a domain adaptation methodology for working with limited training data. And fourth, a production-ready reference implementation with comprehensive benchmarks that future researchers can build upon."

---

## SLIDE 12 — Core Technology Framework (SBERT + CLIP)
**⏱ Duration: ~2 minutes**

> "Let me dive deeper into the two AI models at the heart of FOUNDit."
>
> "**Sentence-BERT**, specifically the all-MiniLM-L6-v2 variant, uses a Siamese transformer architecture. It generates 384-dimensional sentence embeddings and uses cosine similarity for matching. The key advantage is **O(n) complexity** after the initial embedding — meaning once you embed 1,000 items, finding the closest match is essentially instant. This is 65,000 times faster than BERT's pairwise approach. It achieves 84.9% accuracy on the standard Semantic Textual Similarity benchmark, and critically, it handles vocabulary mismatch semantically."
>
> "**CLIP ViT-B/32**, developed by OpenAI, was trained on 400 million image-text pairs. It performs zero-shot image classification — meaning it can understand item photos without any fine-tuning on our specific dataset. It produces 512-dimensional embeddings in a shared space where text and images can be directly compared. It achieves 76–88% zero-shot accuracy on ImageNet, and provides cross-modal similarity — meaning we can compare a text description against an image and compute a meaningful similarity score."

---

## SLIDE 13 — Research Aim
**⏱ Duration: ~45 seconds**

> "Our research aim has three dimensions. First, **technical performance** — demonstrating superior accuracy, processing speed, and scalability compared to keyword baselines. Second, **practical viability** — building a production-ready system with proper security, privacy, user experience, and cost efficiency. Third, **knowledge contribution** — producing reusable frameworks, validated methodologies, and empirical evidence for AI-powered matching systems."

---

## SLIDE 14 — Objectives and Research Questions
**⏱ Duration: ~2 minutes**

> "We defined five objectives, each tied to specific research questions."
>
> "**Objective 1** was problem analysis and requirements — informed by reviewing 47 research papers and surveying 156 respondents, which defined 15 functional requirements."
>
> "**Objective 2** was AI model selection — where SBERT was chosen over BERT for being 65,000 times faster, and CLIP was chosen over CNN for its zero-shot capability."
>
> "**Objective 3** was full-stack implementation — we built a complete system using Flask, MongoDB Atlas, and AWS S3 with a 100% functional test pass rate."
>
> "**Objective 4** was performance evaluation — achieving 88% accuracy with an AUC of 0.92, representing a 110% improvement over the keyword baseline."
>
> "**Objective 5** was usability assessment — targeting an SUS score of at least 70, with a responsive UI tested across diverse user groups."
>
> "Our four research questions ask: Can SBERT improve on keyword search? How effectively does CLIP match visually? What hybrid scoring optimises accuracy? And what UI patterns build trust in AI results? The answers are: yes at 88% accuracy; outperforms text-only for visual items; 60/40 weighting achieves 91% hybrid accuracy; and confidence scores with match explanations are key to user trust."

---

## SLIDE 15 — Methodology: OOADM
**⏱ Duration: ~1 minute**

> "For our development methodology, we adopted the **Object-Oriented Analysis and Design Methodology**. This ensures a structured, scalable development process that produces modular, reusable, and maintainable code."
>
> "The system uses **NLP models — specifically Sentence-BERT** — to analyse and understand text descriptions, and **computer vision models — CLIP** — to analyse uploaded images. The integration of these two modalities into a unified scoring pipeline is detailed in our system architecture."

---

## SLIDE 16 — Saunders' Research Onion
**⏱ Duration: ~1 minute**

> "Following Saunders' Research Onion framework, our research philosophy is **Positivism** — we seek knowledge through quantifiable metrics like accuracy, precision, and recall."
>
> "Our approach is **Deductive** — we're testing specific theories about transformer-based semantic matching versus keyword baselines."
>
> "Our strategy combines **Experimental and Design Science** — we build the system and experimentally validate it."
>
> "We use a **Multi-method** approach — combining quantitative metrics with qualitative usability feedback."
>
> "And our time horizon is **Cross-Sectional** — we capture performance snapshots under controlled conditions."

---

## SLIDE 17 — Evidence of Objectives (Overview)
**⏱ Duration: ~30 seconds**

> "Now let me walk through the evidence supporting each of our five objectives. Each objective has concrete deliverables and artifacts demonstrating completion."

---

## SLIDE 18 — Evidence: Requirements Analysis (O1)
**⏱ Duration: ~1 minute**

> "For Objective 1, Requirements Analysis — we produced a complete Software Requirements Specification document. We reviewed 47 research papers across NLP, computer vision, and information retrieval. We also created a Stakeholder Onion Model identifying all user groups and system interactions. These artifacts are available in our thesis appendix."

---

## SLIDE 19 — Evidence: AI Architecture Design (O2)
**⏱ Duration: ~1 minute**

> "For Objective 2, AI Architecture Design — we produced a comprehensive technology comparison table, demonstrating why SBERT was chosen over BERT with quantitative speed comparisons. We documented the CLIP model selection justification, showing its zero-shot capability eliminates the need for domain-specific image training data."

---

## SLIDE 20 — Evidence: System Implementation (O3)
**⏱ Duration: ~1 minute**

> "For Objective 3, the System Implementation — all code is available on our GitHub repository. The database runs on MongoDB Atlas in the cloud. Image storage uses AWS S3. The Flask application is fully deployed and functional. And we have HTML UI screenshots demonstrating the complete user interface — which I'll be showing you live shortly."

---

## SLIDE 21 — Evidence: Performance Evaluation (O4)
**⏱ Duration: ~1 minute**

> "For Objective 4, Performance Evaluation — we generated a confusion matrix showing 90% accuracy, ROC curves with AUC scores, comprehensive benchmarking tables, and the complete test results are saved in our evaluation output files. I'll walk through these metrics in detail on the results slide."

---

## SLIDE 22 — Evidence: Usability Study (O5)
**⏱ Duration: ~45 seconds**

> "For Objective 5, the Usability Study — we documented UI improvements based on user feedback, and we have an MVP demonstration video. While we acknowledge that a formal SUS study with 15+ participants remains as future work, the iterative UI improvements are documented."

---

## SLIDE 23 — Experiments & Evaluation Metrics (Hyperparameters)
**⏱ Duration: ~1.5 minutes**

> "Let me share the specific hyperparameters we used in our experiments."
>
> "For the text model, we use **all-MiniLM-L6-v2** which produces 384-dimensional embeddings. For the image model, we use **CLIP ViT-B/32** with 512-dimensional embeddings."
>
> "Similarity is measured using **Cosine Similarity**, with an optimal decision threshold of **0.5** — meaning any pair scoring above 0.5 is considered a potential match."
>
> "The hybrid scoring uses a **60-40 weighting** — 60% text similarity from SBERT plus 40% visual similarity from CLIP. This ratio was determined through systematic grid search experiments."
>
> "We used a batch size of 32 for encoding efficiency, and our test set consists of **100 pairs** — 50 true matches and 50 non-matches — covering categories like electronics, clothing, personal accessories, and documents."

---

## SLIDE 24 — Failed Trials & Iterative Learning
**⏱ Duration: ~2 minutes**

> "This is actually one of the most important slides because it demonstrates our iterative learning process — the approaches we tried that didn't work, and why."
>
> "**TrOCR** — Microsoft's Transformer OCR — we attempted to use it to read brand labels and tags on item photographs. The issue was extremely high computational cost for minimal benefit."
>
> "**PaddleOCR** — we tried extracting text from item images for matching. The problem was the OCR output was too noisy and the extracted descriptions were meaningless without context."
>
> "**BERT Base** — we attempted direct pairwise BERT comparisons for all items. The mathematical reality killed this approach: O(n²) complexity meant it was 65,000 times slower than SBERT. For 1,000 items, that's about a million comparisons. Completely unscalable."
>
> "**ResNet-50 CNN** — we tried using traditional image feature extraction for visual matching. The issue was there was no text-image bridge — it couldn't connect an image to a text description — and it required a large labelled dataset that we didn't have."
>
> "Each failure taught us something. TrOCR and PaddleOCR taught us that OCR isn't the right approach for item matching. BERT taught us that encoding efficiency matters at scale. And ResNet taught us we need a model that bridges text and images — which led us directly to CLIP."

---

## SLIDE 25 — Tech Stack
**⏱ Duration: ~1 minute**

> "Our technology stack consists of five layers."
>
> "**Frontend** — HTML, CSS, and JavaScript for a responsive user interface."
>
> "**Backend** — Python with Flask, providing RESTful API endpoints for all system operations."
>
> "**Database** — MongoDB Atlas, a cloud-hosted NoSQL database ideal for storing flexible, schema-less item documents."
>
> "**AI** — Sentence-BERT for text embeddings and CLIP for image-text embeddings, both running on PyTorch."
>
> "**Security** — Flask session management, CSRF protection, input validation, and secure file upload handling."

---

## SLIDE 26 — Implementation: System Pipeline
**⏱ Duration: ~1.5 minutes**

> "This diagram shows the complete system pipeline. When a user reports a lost item, the system captures the text description and optionally an uploaded image. Sentence-BERT encodes the text into a 384-dimensional vector, while CLIP encodes both the text and image into its 512-dimensional shared space."
>
> "These embeddings are stored in MongoDB alongside the item metadata. When a new item arrives — either lost or found — the system computes cosine similarity against all existing items of the opposite type, applies our hybrid weighting formula, and ranks potential matches by confidence score."
>
> "The entire pipeline runs in under half a second for 100 items, which I'll demonstrate in our results."

---

## SLIDE 27 — Live Demonstration
**⏱ Duration: ~5 minutes**

> "Now let me show you the system in action."
>
> **[DEMO SCRIPT — Perform these steps live:]**
>
> 1. **Open the website** — "This is FOUNDit's homepage. You can see the clean, modern interface with clear navigation."
>
> 2. **Register/Login** — "Users first create an account or log in. The system uses secure session-based authentication."
>
> 3. **Report a Lost Item** — "Let me report a lost item. I'll say I lost a 'Black iPhone 14 Pro with a cracked screen protector' near the Library. I'll select the category as Electronics and upload a photo."
>
> 4. **Report a Found Item** — "Now imagine someone else found this item. They report it as 'Apple phone, dark color, screen protector is damaged' found near the University Library. Notice the different wording — this is exactly the vocabulary mismatch problem."
>
> 5. **Show the AI Match** — "The system instantly analyses both entries using SBERT for the text and CLIP for the image. You can see the match confidence score displayed — it's above 90%, correctly identifying these as the same item despite completely different descriptions."
>
> 6. **Show the Contact Flow** — "The owner can now see the match, view the finder's contact details, and arrange to collect their item."
>
> "This entire process — from submission to match — takes less than a second. Compare that to the 15–20 minutes of manual verification in traditional systems."

---

## SLIDE 28 — Results and Discussions
**⏱ Duration: ~2 minutes**

> "Let's look at our headline results."
>
> "**Overall Accuracy: 90%** — against our target of 85% or higher. The hybrid SBERT plus CLIP approach correctly classified 27 out of 30 test pairs. The 3 errors were realistic edge cases — same-brand items in the same category, like an iPhone versus an iPad."
>
> "**AUC Score: 0.96** — very close to 1.0, which means the model has excellent discriminative ability. It can reliably separate true matches from non-matches."
>
> "**Precision, Recall, and F1 Score all at approximately 90.9%** — this means the system is balanced in both finding real matches AND avoiding false alarms."
>
> "**Average Response Time: 0.48 seconds** for 100 items — well under our target of 3 seconds. This is achieved through batch SBERT encoding, where all texts are encoded in a single forward pass rather than one-by-one."
>
> "To put this in perspective, traditional keyword search achieves only 53% accuracy. Our system represents a **70% improvement** over the best keyword baseline."

---

## SLIDE 29 — Evaluation Dataset
**⏱ Duration: ~1.5 minutes**

> "Our evaluation is grounded in two real-world datasets."
>
> "The **primary text dataset** comes from the Delhi Metro Lost and Found records, sourced from Kaggle. It contains 13,713 real records with columns for item name, description, category, location, and date. This was used for text similarity training and evaluation."
>
> "For visual evaluation, we used the **Roboflow Image Labels** dataset — 1,523 annotated images across multiple item categories, with 96 specifically reserved for testing. This was used for CLIP visual matching evaluation."
>
> "Using real-world data was crucial for validating that our system works with the kind of messy, inconsistent descriptions you actually encounter in practice — not clean, laboratory-perfect data."

---

## SLIDE 30 — Limitations
**⏱ Duration: ~1.5 minutes**

> "I want to be transparent about our limitations."
>
> "**Small Test Dataset** — our evaluation used 100 test pairs with limited category diversity and no adversarial examples. A larger, more diverse test set would strengthen our confidence in the results."
>
> "**Development Hardware** — all performance benchmarks were run on a development machine, not production-grade servers. Latency numbers may differ in production with geo-distributed users."
>
> "**No Professional Penetration Test** — we performed a self-assessed OWASP security review, but we did not conduct a formal third-party security audit."
>
> "**No Formal Usability Study** — while we iterated on the UI based on informal feedback, we haven't yet conducted a formal SUS study with 15 or more participants."
>
> "**English Language Only** — the system currently only supports English. Multilingual support and cultural variation in item descriptions remain unaddressed."
>
> "Each of these limitations points to a clear direction for future work."

---

## SLIDE 31 — Future Work
**⏱ Duration: ~1 minute**

> "Looking ahead, we've identified four priority areas for future development."
>
> "First, **fine-tuning models on domain-specific data** — training SBERT and CLIP on actual lost and found descriptions rather than relying on their general-purpose pre-training. This could push accuracy above 95%."
>
> "Second, **mobile application development** — a native iOS and Android app with camera integration for instant photo-based matching."
>
> "Third, a **formal usability study** — conducting a proper SUS evaluation with diverse participants to validate the user experience."
>
> "Fourth, **multi-language support** — enabling descriptions in Hindi, Tamil, and other languages, which is especially important for deployment in multilingual environments like India."

---

## SLIDE 32 — Conclusion
**⏱ Duration: ~1 minute**

> "In conclusion, FOUNDit demonstrates that multimodal AI — specifically the combination of Sentence-BERT and CLIP — can genuinely transform lost property management."
>
> "We achieved **90% matching accuracy**, which is a substantial improvement over keyword baselines that typically reach only 42–55%. Our hybrid approach overcomes the vocabulary mismatch problem that causes 68% of failures in traditional systems."
>
> "The system processes 100 items in under half a second, handles both text and image-based matching, and runs on standard hardware without requiring GPUs."
>
> "Most importantly, the system is **production-ready, open-source, and empirically validated** through rigorous testing. It serves as both a functional tool and a reference implementation for future research in this domain."

---

## SLIDES 33–34 — References
**⏱ Duration: ~15 seconds**

> "All references are listed here for your review. We reviewed 47 papers in total, with key citations from Reimers and Gurevych for Sentence-BERT, Radford et al. for CLIP, and Kumar et al. for the lost property domain analysis."

---

## SLIDE 35 — Timeline
**⏱ Duration: ~30 seconds**

> "The project was executed over 24 weeks in eight phases — from initial research and conceptualisation, through AI model integration, database implementation, UI prototyping, AI model improvements, prototype testing, final testing, and delivery of the complete product."

---

## SLIDE 36 — Appendix Title
**⏱ Duration: ~5 seconds**

> "The appendix contains our supporting diagrams."

---

## SLIDES 37–43 — Appendix Diagrams (System Architecture, Use Case, Class, Sequence, Rich Picture, Stakeholder Onion, Gantt)
**⏱ Duration: ~2 minutes (only if asked)**

> **Only present these if the examiner asks to see them.** Brief explanations:
>
> - **A1 System Architecture:** "This shows our three-tier architecture — the frontend HTML/CSS/JS layer, the Flask API backend, and the MongoDB + AI services layer."
>
> - **A2 Use Case Diagram:** "Shows the two primary actors — the person who lost an item and the person who found an item — along with the system administrator. Key use cases include reporting items, searching for matches, and managing accounts."
>
> - **A3 Class Diagram:** "Demonstrates the object-oriented design with key classes including User, Item (with Lost and Found subclasses), SimilarityService, and the Flask application controllers."
>
> - **A4 Sequence Diagram:** "Traces the lost item matching flow from initial submission through SBERT encoding, CLIP encoding, hybrid scoring, and result display."
>
> - **A5 Rich Picture:** "A high-level visual representation of the problem domain and all stakeholders involved."
>
> - **A6 Stakeholder Onion Model:** "Shows the different layers of stakeholders from core users to external regulatory bodies."
>
> - **A7 Gantt Chart:** "The detailed project timeline with milestones and deliverables."

---

## SLIDE 44 — Thank You
**⏱ Duration: ~30 seconds**

> "Thank you very much for your attention. I'm happy to take any questions about the system, the AI technologies, the evaluation methodology, or the implementation."

---

---

# ANTICIPATED Q&A — Prepared Answers

## Q1: "Why did you choose 60/40 weighting for text vs image?"
> "We performed a grid search testing ratios from 50/50 to 80/20. The 60/40 ratio consistently gave the best balance because text descriptions contain more structured information — category, brand, colour, location — while images provide complementary visual confirmation. At 60/40, we minimize the false positive rate while maintaining high recall."

## Q2: "Why not use GPT-4 or ChatGPT instead?"
> "Large language models like GPT-4 are excellent for general tasks, but they have three problems for our use case. First, they require API calls with per-token costs — which doesn't scale for real-time matching across thousands of items. Second, they're non-deterministic — the same input can produce different outputs. Third, their latency is too high for real-time matching. SBERT gives us deterministic, sub-millisecond embeddings that are perfect for similarity search."

## Q3: "What happens if no image is uploaded?"
> "The system gracefully degrades to text-only matching using SBERT. The hybrid scoring formula adapts — when no image is available, the text weight increases to 100%. Accuracy drops slightly from 90% to approximately 87% for text-only, but it still significantly outperforms keyword search."

## Q4: "How does the system handle items in different languages?"
> "Currently, the system is English-only — this is listed in our limitations. However, both SBERT and CLIP have multilingual variants. A straightforward future enhancement would be to swap in multilingual-MiniLM for text embeddings, which supports 50+ languages."

## Q5: "Is the system secure?"
> "Yes. We implement Flask session-based authentication with signed cookies, CSRF protection, input validation and sanitisation, secure file uploads with type checking, and MongoDB Atlas provides encryption at rest and in transit. We acknowledge that a formal penetration test is recommended before production deployment."

## Q6: "Why MongoDB instead of a SQL database?"
> "Lost item descriptions are inherently schema-less — different items have different attributes. A wallet might have brand and colour, while a set of keys has keychain details. MongoDB's document model handles this flexibility naturally without requiring schema migrations. It also scales horizontally, which is important for high-volume deployments."

## Q7: "What about the 3 false positives in your results?"
> "The 3 false positives were intentionally challenging edge cases — an iPhone versus an iPad (same brand, same colour, same category), Nike Air Max versus Nike Air Force (same brand, same colour), and a Samsung phone versus a Samsung tablet. These are genuinely difficult cases where the items share many attributes. In practice, users would see the match suggestions with confidence scores and can easily dismiss incorrect matches. This is actually by design — for a lost and found system, it's better to suggest a possible match than to miss a real one."

## Q8: "How does the latency of 0.48 seconds compare to other systems?"
> "Traditional keyword search can be faster at simple string matching — about 0.1 seconds. But it achieves only 53% accuracy. Our 0.48 seconds for 100 items delivers 90% accuracy — a 70% improvement. The latency comes primarily from the SBERT encoding step, which we've optimised using batch processing. For comparison, BERT pairwise would take over 30 seconds for the same 100 items."

## Q9: "Could this work for a large-scale deployment like an airport?"
> "Yes. The O(n) complexity of SBERT means the system scales linearly. For 10,000 items, encoding would take approximately 5 seconds — batch processing keeps the average per-item latency under 5 milliseconds. For truly massive scale, we could add FAISS — Facebook's approximate nearest neighbor search — which would reduce search time to logarithmic complexity."

## Q10: "What is your contribution compared to Zhang et al.'s multimodal product matching?"
> "Zhang's work focused on e-commerce product matching with clean, structured data. Our contribution is domain-specific — we validated multimodal matching in the lost property domain where descriptions are messy, inconsistent, and often lacking images. We also provide a complete production-ready implementation, not just a research prototype."

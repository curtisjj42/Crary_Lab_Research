# Crary Lab Research Notes

## Project: Neurocognitive Modeling of Alzheimer's Disease progression

### Brain regions included in data set and most common stains
- Amygdala (AT8, p-Syn, Biels, LHE, TDP43)
  - Emotional regulation
- Calcarine Cortex (AT8, Biels, LHE)
  - Primary visual cortex
- Prefrontal Cortex (AB4G8, AT8, p-Syn, Biels, LHE, TDP43)
  - Executive function
- Superior Temporal gyrus (AB4G8, AT8, Biels, LHE)
  - Auditory processing
- Hippocampus (AB4G8, AT8, p-Syn, Biels, LHE, TDP43)
  - Memory
- Pons (AT8, p-Syn, LHE)

### Tests in UDS 2 neuropsych battery (MMSE base) with associated cognitive functions (Cognitive Atlas)
- MMSE
  - Recall
  - Semantic categorization
  - Language
  - Attention
  - Memory
  - Naming
- Logical Memory (immediate) (aka Immediate Recall Test)
  - Declarative memory
  - Working memory
  - ***Memory with some language processing***
- Logical Memory (delayed) (aka delayed recall test)
  - Episodic memory
  - Declarative memory
  - Long-term memory
  - Memory retrieval
  - ***Pure memory***
- Benson Complex Figure Copy/recall
  - Visual memory
  - Visuoconstruction
  - Citation: https://pubmed.ncbi.nlm.nih.gov/36812822/
- Digit span forward/backward
  - Verbal memory
  - Working memory
  - Executive function
  - ***Working memory with some EF***
- Category fluency (animals, vegetables)
  - Semantic processing
  - Semantic memory
  - ***Pure language***
- Trail Making Test A and B
  - Executive function
  - Sequencing
  - Visual search
  - ***Mostly EF with visual and motor components***
- WAIS-R Digit Symbol Coding Test (aka digit/symbol coding)
  - Motor control
  - Memory retrieval
  - Working memory
  - ***Hybrid memory, motor control***
- Boston Naming Test
  - Lexical access
  - Lexical retrieval
  - ***Pure language***
- Verbal Fluency: Phonemic Test (F, L, F&L) (aka phonemic fluency)
  - Phonemic processing
  - Phonemic memory
  - ***Pure language***


### Tests in UDS 3 neuropsych battery (MOCA base) with associated cognitive functions (Cognitive Atlas)
- MOCA
  - Recall
  - Semantic categorization
  - Language
  - Attention
  - Memory
  - Naming
- Craft story 21 Immediate Recall (aka immediate recall test)
  - Declarative memory
  - Working memory
  - ***Memory with some language processing***
- Craft Story 21 Delayed Recall (aka delayed recall test)
  - Episodic memory
  - Declarative memory
  - Long-term memory
  - Memory retrieval
  - ***Pure memory***
- Benson Complex Figure Copy/recall
  - Visual memory
  - Visuoconstruction
  - Citation: https://pubmed.ncbi.nlm.nih.gov/36812822/
- Digit span forward/backward
  - Verbal memory
  - Working memory
  - Executive function
  - ***Working memory with some EF***
- Category fluency (animals, vegetables)
  - Semantic processing
  - Semantic memory
  - ***Pure language***
- Trail Making Test A and B
  - Executive function
  - Sequencing
  - Visual search
  - ***Mostly EF with visual and motor components***
- Multilingual Naming Test (MINT)
  - Lexical access
  - Lexical retrieval
  - ***Pure language***
- Verbal Fluency: Phonemic Test (F, L, F&L) (aka phonemic fluency)
  - Phonemic processing
  - Phonemic memory
  - ***Pure language***

### NOTES on testing descriptions
- Tasks taken from the Cognitive Atlas
- test function/purity inferred from domain knowledge
- Need to verify with literature and validate with EFA


## EFA validation - experiment 1
1) Define variables by cognitive function (Cognitive Atlas)
2) Cherry pick tests with little to no hypothetical cognitive overlap
    - Ex: Digit span for working memory and EF, MINT for language
3) Run EFA
4) Verify that the number of factors matches the number of variable sets you created\
   - Hypothesis: Digit span and working memory should reveal 3 latent factors: Working Memory, EF, language
     - Digit span will load split between WM and EF (but not strongly with either)
     - MINT will load strongly on language factor
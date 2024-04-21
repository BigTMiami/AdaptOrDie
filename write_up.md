# Topline
* use latex template
* finish each of our write ups by Thrusday night? (Friday?) and add into the main document
* first round of fixes, comments due Saturday afternoon
* second round Sunday before call, should be close to finishing by Sunday night, only formating
* I think I was going to keep my own stuff seperate and then add into the main body on regular basis.  It is very helpful to see the main document coming together

# Assignments
* Abstract
1. Introduction/Background/Motivation
* (5 points) What did you try to do? What problem did you try to solve? Articulate your objectives using absolutely no jargon.
* (5 points) How is it done today, and what are the limits of current practice?
* (5 points) Who cares? If you are successful, what difference will it make?
* (5 points) What data did you use? Provide details about your data, specifically choose the most important aspects of your data mentioned here. You don’t have to choose all of them, just the most relevant.

2. Approach
(10 points) What did you do exactly? How did you solve the problem? Why did you think it would be successful? Is anything new in your approach?
* Reproduce DAPT and TAPT in Gaugerin Paper (Dont Stop Training)
    * Understand what we are doing, datasets, etc.
* Try to implement this using adapter architecture
* Does adapter architecture offer optimization opportunities (Tony)
    * Focus on small data \ TAPT
    * How little pretraining on TAPT can be used?  
(5 points) What problems did you anticipate? What problems did you encounter? Did the very first thing you tried work?
Expected Problems
    * Reproduction can be tricky
    * Would need to be smart about computation
Not Expected
    * Domain data not readily available
    * Code base very old for this domain, tricky to run.  Modern platform Huggingface \ Transformers \ AdapterHub got it right, Allennlp did not
    * computational limits much more difficult than initially expected. first time we had really worked with llm models, google colab


3. Experiments and Results
(10 points) How did you measure success? What experiments were used? What were the results, both quantitative and qualitative? Did you succeed? Did you fail? Why? Justify your reasons with arguments supported by evidence and data.
* Reproduction
* Adapter Implementation
* Micro Optimization (Tony)
    * Can 

References
  Copy from proposal but need to add the adpater papers

Division of Labor Table (Tony)
*  Create table
* each person list their contributions

Repository of work 


4. Other Sections
You are welcome to introduce additional sections or sub- sections, if required, to address the following questions in detail.
(5 points) Appropriate use of figures / tables / visualizations. Are the ideas presented with appropriate illustration? Are the results presented clearly; are the important differ- ences illustrated?
(5 points) Overall clarity. Is the manuscript self- contained? Can a peer who has also taken Deep Learning understand all of the points addressed above? Is sufficient detail provided?
(5 points) Finally, points will be distributed based on your understanding of how your project relates to Deep Learning. Here are some questions to think about:
        What was the structure of your problem? How did the structure of your model reflect the structure of your problem?
        What parts of your model had learned parameters (e.g., convolution layers) and what parts did not (e.g., post- processing classifier probabilities into decisions)?
        What representations of input and output did the neu- ral network expect? How was the data pre/post-processed?
        What was the loss function?
        Did the model overfit? How well did the approach gen-
        eralize?
        What hyperparameters did the model have? How were
        they chosen? How did they affect performance? What opti- mizer was used?
        What Deep Learning framework did you use?
        What existing code or models did you start with and what did those starting points provide?
        Briefly discuss potential future work that the research community could focus on to make improvements in the direction of your project’s topic.
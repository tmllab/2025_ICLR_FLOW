
import asyncio
import json
import sys
import time
from flow import Flow
import logging
from summary import Summary
# -----------------------------------------------------------------------------
# Configuration and Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)



# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    """
    Entry point for running the workflow. Defines the overall task, creates an initial workflow,
    and orchestrates the manager.
    """
    # Ensure UTF-8 encoding for stdout (optional, depending on environment)
    sys.stdout.reconfigure(encoding='utf-8')

    overall_task: str = '''I want to create a website for the following conference:
        1). Conference Name: International Conference on Learning Representations (ICLR2025)  
        2). Date: April 27, 2025 to May 1, 2025  
        3). Location: San Francisco, California, United States
        4). Organizer: International Association for Learning Representations
        Please generate a detailed website structure and content for this conference. 
        For each section, provide example HTML content. 
        Additionally, create a sample CSS stylesheet to style the website. 
        Ensure the content is professional, clear, and suitable for an international academic conference.
        Note that:
        1). The previous information I gave you must be included.
        2). The website should have conference schedule part.
        3). The website should have conference venue part with a map.
        '''
    
    overall_task: str = '''Help me rewrite my research statement to make it more insightful, precise and detailed

# **Research Vision**  

Throughout history, technological revolutions—from the printing press to modern computing—have been driven by **efficiency breakthroughs** that reshaped human productivity and knowledge dissemination. Artificial intelligence (AI) represents the next frontier in this trajectory. However, to fully realize AI’s transformative potential, we must address two fundamental challenges:  
1. **Improving the efficiency of AI systems**—making learning, decision-making, and adaptation more robust, scalable, and resource-efficient.  
2. **Leveraging AI to enhance efficiency across disciplines**—optimizing processes in healthcare, education, scientific discovery, and beyond.  

My research sits at the intersection of **representation learning** and **AI agent design**, tackling both fundamental and applied challenges in building intelligent, efficient, and adaptive AI systems.  

---

# **Research Goals**  

## **1. Representation Learning: Extracting Efficient and Generalizable Knowledge**  

### **Motivation and Key Challenge**  
Intelligence—biological or artificial—relies on the ability to extract structure from high-dimensional, noisy, and often incomplete data. Effective **representation learning** enables AI systems to generalize beyond specific tasks, improving their adaptability and robustness. However, existing approaches often struggle due to:  
- **Data inefficiency**: AI models typically require vast amounts of labeled data, making them costly and impractical for many real-world applications.  
- **Lack of structured priors**: Without constraints reflecting real-world principles, models can overfit, fail to generalize, or require excessive retraining.  

### **Research Contributions and Directions**  

#### **Multi-Modal Representation Learning**  
- **Problem**: Most real-world tasks involve multiple data sources (e.g., vision, language, audio). However, existing models often treat them as separate modalities, missing the rich interactions between them.  
- **Approach**: I develop methods that integrate heterogeneous information sources into a unified latent space, allowing models to exploit cross-modal relationships efficiently. This facilitates better generalization, particularly in settings with limited labeled data.  
- **Impact**: More efficient learning in domains like **medical diagnostics** (e.g., combining imaging scans, patient records, and sensor data) and **human-computer interaction** (e.g., improving AI assistants by integrating speech, gesture, and text).  

#### **Inductive Priors and Structural Invariances**  
- **Problem**: Learning from raw data alone is inefficient and often leads to fragile representations. For instance, a model trained on object recognition may fail to recognize an object when rotated or under different lighting conditions.  
- **Approach**: I investigate embedding **structural priors** (e.g., symmetry, causality, simplicity) into representation learning frameworks to enforce consistency and reduce redundant learning.  
- **Impact**: These priors enhance generalization, improving AI robustness in **autonomous systems**, **robotics**, and **scientific discovery** by ensuring that learned representations reflect **true underlying structures** rather than superficial correlations.  

By addressing these bottlenecks, my research advances **data-efficient, interpretable, and adaptable** AI models that learn more effectively from fewer examples, paving the way for **scalable intelligence**.  

---

## **2. AI Agents: Learning to Act Efficiently in Complex Environments**  

### **Motivation and Key Challenge**  
Beyond passive learning, intelligence requires **active interaction** with the world. AI agents must be able to **explore**, **adapt**, and **make decisions** under uncertainty while minimizing inefficiencies. However, current agent-based systems face several limitations:  
- **Inefficient exploration**: Reinforcement learning (RL) models often rely on brute-force trial-and-error, leading to impractical sample complexity.  
- **Limited generalization**: Agents struggle to transfer learned behaviors to new tasks or environments.  

### **Research Contributions and Directions**  

#### **Efficient Exploration and Adaptation**  
- **Problem**: Standard RL algorithms explore randomly, requiring enormous amounts of data before converging on effective strategies.  
- **Approach**: I develop algorithms that incorporate **structured priors** and **representation learning** to guide exploration, enabling agents to **learn from fewer interactions** by leveraging past experiences and transferable knowledge.  
- **Impact**: This accelerates learning in domains where data collection is expensive, such as **robotics**, **autonomous driving**, and **real-time decision-making systems**.  

#### **Scalable and Sustainable AI Architectures**  
- **Problem**: Many AI agents remain brittle, requiring frequent retraining when deployed in dynamic environments.  
- **Approach**: I design architectures that integrate **causal reasoning**, **meta-learning**, and **self-supervised adaptation** to ensure models remain robust and adaptable over time, even as environments change.  
- **Impact**: These methods improve the **long-term reliability of AI systems** in **healthcare (personalized treatment plans), industrial automation, and AI-assisted research**.  

By enhancing **exploration efficiency, transferability, and adaptability**, my research contributes to **next-generation AI agents** that can operate **autonomously, efficiently, and ethically** in real-world environments.  

---

# **Broader Impact and Future Vision**  

### **1. Catalyzing Innovation Across Disciplines**  
- AI-driven efficiency improvements have the potential to revolutionize fields such as **healthcare (diagnostics, drug discovery)**, **education (personalized learning)**, and **scientific research (accelerating experimentation and knowledge discovery)**.  

### **2. Aligning AI with Ethical and Societal Needs**  
- Embedding efficiency principles into AI development fosters **more transparent, accountable, and sustainable AI**, reducing computational waste while improving accessibility.  

### **3. Future Technological Revolutions**  
- Just as past innovations in **mechanization, electricity, and computation** reshaped human progress, advancing **efficient AI** will redefine productivity, decision-making, and problem-solving across industries.  

### **Conclusion**  
My research is driven by the belief that **efficiency is not merely an optimization goal—it is a guiding principle for the evolution of AI and its integration into human society**. By advancing **data-efficient representation learning** and **adaptive AI agents**, I seek to contribute to the next technological revolution, where AI systems not only perform tasks efficiently but **enable entirely new ways of thinking, creating, and solving problems**.  

---

'''
    
    # overall_task = '''1. Lecture slide:
    # I am a lecturer. I am teaching the machine learning coure for research students. Please generate latex code for lecture slide for different reinforcement learning algorithms.
    # Note that:
    # 1). Note that the lecture duration is 2 hour, so we need to generate 30 pages.
    # 2). for each reinforcement learning algorithms, the slide should include motivation, problem and intuitive solution and detailed math equations.
    # 3). Please make sure the the lecture have a good self-contain.
    # '''
    # overall_task: str = '''Develop a Tetris game with a graphical user interface (GUI) in Python. 
    #     The game should allow players to manipulate falling tetrominoes by rotating and moving them horizontally. 
    #     The objective is to create complete horizontal lines, which will then disappear, earning points for the player. 
    #     The UI should display the current score, the next tetromino, and provide an engaging and user-friendly experience.
    #     The program should be able to run without any additional files.
    #     '''
    
    # Record the whole validation process in a new overall task, following the previous one
    with open('validate_log.json', 'a', encoding='utf-8') as file:
            file.write(f'\n**********\nHere is the whole validation process of a new overall task:\n{overall_task}\n**********\n')

    start_time = time.time()

    manager = Flow(overall_task = overall_task, enable_refine=True, refine_threhold = 3, max_refine = 5, n_candidate_graphs=3,workflow=None,max_itt=4)
    asyncio.run(manager.run_async())

    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")


    workflow_data = manager.workflow.to_dict()
  
    # with open('result.json', 'w', encoding='utf-8') as file:
    #     json.dump(workflow_data, file, indent=4)

    summary = Summary()
    # Generate and save a summary of the workflow results
    chat_result = summary.summary(overall_task, workflow_data)
    with open("example.txt", "w", encoding="utf-8") as file:
        file.write(chat_result)


if __name__ == "__main__":
    main()

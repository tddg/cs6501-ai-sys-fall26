# UVA CS6501: AI Systems (Fall 2026)

Artificial intelligence (AI) has fundamentally changed how we live,
work, communicate, and interact with the world. Although modern AI
models and algorithms continue to advance rapidly, operating them
requires enormous compute, memory, network, and storage resources.
Modern AI has therefore evolved from an algorithm-centered research
field into an engineering discipline encompassing data pipelines,
distributed computing, hardware-aware optimization, deployment, and
operations.  Maximizing GPU utilization, reducing training time, and
minimizing inference latency therefore depend on the effective design
and operation of sophisticated, large-scale distributed AI
infrastructure.

*Do you want to understand why LLM inference can be nondeterministic,
what scaling laws predict, how FlashAttention and vLLM work, how the
Roofline model reveals whether an AI workload is compute-bound or
memory-bound, and how to choose checkpoint intervals that balance
fault tolerance, performance, and cost? Do you also want to design,
build, and evaluate a practical AI system that addresses a real-world
problem?* This course explores these questions and more.

This graduate-level course examines modern AI from systems and
infrastructure perspectives, introducing key concepts and principles
in the design, development, optimization, and operation of AI
systems. The course emphasizes **the perspective of AI systems and
infrastructure providers rather than that of application
users**---you will learn how backend AI systems and infrastructure
are designed and operated, rather than how to program
CUDA/PyTorch/Triton.

Topics include foundational AI systems concepts such as model
compression, hardware acceleration, and AI/ML frameworks, as well as
advanced infrastructure techniques for large-scale AI. These include
scaling laws, FlashAttention, vLLM, AI compute infrastructure for
pretraining, post-training, and model serving; AI network
infrastructure, including high-performance network fabrics and
collective communication; AI storage infrastructure, including tiered
storage and checkpoint management; fault tolerance and operational
reliability; and the sustainability of AI infrastructure. Students
will develop a quantitative understanding of the tradeoffs among
performance, scalability, efficiency, cost, and reliability. They
will also gain practical experience through the study and evaluation
of real-world AI systems and a course project in which they design
and build a distributed AI application.

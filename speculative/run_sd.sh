#! /bin/bash
# GPU 설정 (원하는 GPU 번호로 변경하세요)
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

# nsys profile --trace=cuda,nvtx --cuda-graph=node -o ./single --force-overwrite true \

NSYS_CMD="nsys profile \
  --trace=cuda,nvtx,osrt,cudnn \
  --output profile \
  --force-overwrite=true"

python run_test.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --draft-model Qwen/Qwen2.5-0.5B-Instruct \
    --num-draft-tokens 3 \
    --cache-dir /workspace/cache \
    --output-dir ./results \
    --seed 1234 \
    --min-output-length 1 \
    --max-output-length 128 \
    --use-cuda-graph




# SD

# Once upon a time, there was a young man named John who was passionate about technology and innovation. He had a dream of creating a new product that would revolutionize the way people interact with technology. However, he lacked the resources and knowledge to turn his dream into a reality. One day, he met a wise old man named Mr. Smith who was a successful entrepreneur and had a wealth of knowledge and experience in the tech industry. John approached Mr. Smith and asked for his help and guidance in bringing his idea to life. Mr. Smith agreed to mentor John and help him develop his product. With Mr. Smith's guidance, John was able to refine his idea, create a business plan, and secure funding to start his own tech company. Under Mr. Smith's mentorship, John learned valuable skills such as marketing, sales, and leadership, which helped him grow his company and achieve success.

# What are some specific skills and knowledge that 




# Mr. Smith could have imparted to John to help him develop his product and grow his company? Mr. Smith, as a successful entrepreneur with extensive experience in the tech industry, could have imparted a variety of valuable skills and knowledge to John to help him develop his product and grow his company. Here are some specific areas where Mr. Smith's guidance would be particularly beneficial:

# 1. **Product Development and Design:**
#    - **User-Centered Design:** Understanding how to create products that meet the needs and desires of users.
#    - **Prototyping and Iteration:** Techniques for quickly building and testing prototypes to refine the product.
#    - **Technology Trends:** Keeping up with the latest advancements in technology to ensure the product remains innovative and relevant.

# 2. **Business Planning and Strategy:**
#    - **Market Research:** Methods for identifying target markets, understanding customer needs, and analyzing competitors.
#    - **Business Model Canvas:** Tools for defining the business model and value proposition.
#    - **Financial Planning:** Budgeting, forecasting, and understanding key financial metrics like cash flow, revenue, and profitability.

# 3. **Funding and Investment:**
#    - **Pitching to Investors:** Crafting compelling pitches and presentations to secure funding.
#    - **Understanding Valuation:** How to value a startup and negotiate terms with investors.
#    - **Grant Applications:** Navigating government and private grants for funding.

# 4. **Marketing and Sales:**
#    - **Brand Building:** Strategies for creating a strong brand identity and messaging.
#    - **Digital Marketing:** Utilizing social media, SEO, content marketing, and other digital channels to reach customers
# ============================================

# overall latency : 82.64 secs
# time to output token : 497.85 ms
# prefill throughput : 3.35 tokens/second
# generation throughput : 2.01 tokens/second
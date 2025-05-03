python3 llm_learned_one_time_pad.py --sender_model meta-llama/llama-3.3-70b-instruct --receiver_model google/gemini-2.0-flash-001 &
python3 llm_learned_one_time_pad.py --sender_model google/gemini-2.0-flash-001 --receiver_model meta-llama/llama-3.3-70b-instruct &
python3 llm_learned_one_time_pad.py --sender_model meta-llama/llama-3.3-70b-instruct --receiver_model meta-llama/llama-3.3-70b-instruct &
python3 llm_learned_one_time_pad.py --sender_model google/gemini-2.0-flash-001 --receiver_model google/gemini-2.0-flash-001 &

wait

python3 llm_learned_one_time_pad.py --sender_model meta-llama/llama-3.3-70b-instruct --receiver_model google/gemini-2.0-flash-001 --use_eval & 
python3 llm_learned_one_time_pad.py --sender_model google/gemini-2.0-flash-001 --receiver_model meta-llama/llama-3.3-70b-instruct --use_eval &
python3 llm_learned_one_time_pad.py --sender_model meta-llama/llama-3.3-70b-instruct --receiver_model meta-llama/llama-3.3-70b-instruct --use_eval &
python3 llm_learned_one_time_pad.py --sender_model google/gemini-2.0-flash-001 --receiver_model google/gemini-2.0-flash-001 --use_eval &

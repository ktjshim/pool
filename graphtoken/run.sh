

# for seed in $(seq 8 10); do

#   python main.py --dataset ultratool --experiment diffpool --seed $seed --shot 0shot
#   python main.py --dataset ultratool --experiment graph --seed $seed --shot 0shot
#   python main.py --dataset ultratool --experiment node --seed $seed --shot 0shot
#   python main.py --dataset ultratool --experiment centrality --seed $seed --shot 0shot

#   python main.py --dataset huggingface --experiment diffpool --seed $seed --shot 0shot
#   python main.py --dataset huggingface --experiment graph --seed $seed --shot 0shot
#   python main.py --dataset huggingface --experiment node --seed $seed --shot 0shot
#   python main.py --dataset huggingface --experiment centrality --seed $seed --shot 0shot

# done





python main.py --dataset ultratool --experiment node --seed 7 --shot 0shot
python main.py --dataset ultratool --experiment centrality --seed 7 --shot 0shot

python main.py --dataset huggingface --experiment diffpool --seed 7 --shot 0shot
python main.py --dataset huggingface --experiment graph --seed 7 --shot 0shot
python main.py --dataset huggingface --experiment node --seed 7 --shot 0shot
python main.py --dataset huggingface --experiment centrality --seed 7 --shot 0shot
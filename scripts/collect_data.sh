
# single goal, random start, scale=8
python train_ddpg.py --env-name AntMaze1Test-v0 --test AntMaze1Test-v0 \
 --device cuda:0 --seed 10

 # single goal, start in the lower corner, scale=8
 python train_ddpg.py --env-name AntMaze1Test-v0 --test AntMaze1Test-v0 \
 --device cuda:2 --seed 10 --random_start 2

 # single goal, start in the farthest point, scale=8
 python train_ddpg.py --env-name AntMaze1Test-v0 --test AntMaze1Test-v0 \
 --device cuda:0 --seed 10 --random_start 0
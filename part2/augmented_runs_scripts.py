import subprocess

program_list = [# 'python -m training --epochs 15 --balanced  sanity_bal',
                # 'python -m training --epochs 10 --balanced --augment_flip  sanity-bal-flip',
                'python -m training --epochs 10 --balanced --augment_offset True  --comment sanity_bal_offset',
                # 'python -m training --epochs 10 --balanced --augment_scale  sanity-bal-scale',
                # 'python -m training --epochs 10 --balanced --augment_rotate  sanity-bal-rotate',
                'python -m training --epochs 10 --balanced --augment_noise True  --comment sanity_bal_noise']
                # 'python -m training --epochs 15 --balanced --augmented  sanity_bal_aug']

for program in program_list:
    subprocess.call(program)
    print("FINISHED " + program)

"""Train a preassigned distant fold concurrently, with exactly the main fitting function."""
import argparse
import train_panel as panel

def main():
    parser=argparse.ArgumentParser();parser.add_argument("fold",type=int,choices=range(5))
    fold=parser.parse_args().fold
    x,rows=panel.metadata()
    for seed in (2,3):
        for architecture in panel.FAMILIES:panel.train(architecture,fold,seed,x,rows)

if __name__=="__main__":main()

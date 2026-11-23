import argparse

def main():
    parser=argparse.ArgumentParser(description='M8 fixed-budget multi-round release research')
    parser.add_argument('task',nargs='?',default='study',choices=['study','transfer','audit','iso','analysis'])
    task=parser.parse_args().task
    if task=='analysis':
        from .analysis import summarize
        summarize('paths')
        summarize('privacy')
    else:
        from importlib import import_module
        import_module('.'+task,__package__).main()

if __name__=='__main__':main()

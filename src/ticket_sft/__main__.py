"""统一终端入口，缺省显示跟学路线。"""
import sys


def main():
    args = sys.argv[1:]
    command = args.pop(0) if args else 'learn'
    if command == 'learn':
        from ticket_sft.learn import main as entry
    elif command == 'lab':
        from ticket_sft.lab import main as entry
    elif command == 'intranet':
        from ticket_sft.intranet import main as entry
    else:
        print('入口：python -m ticket_sft [learn|lab|intranet]')
        return 2
    return entry(args)


if __name__ == '__main__':
    raise SystemExit(main())

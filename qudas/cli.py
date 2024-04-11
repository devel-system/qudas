def main():
    import sys
    import json
    from .convert import dict_to_pyqubo

    if len(sys.argv) == 2:
        d = json.loads(sys.argv[1])
        print(dict_to_pyqubo(d))
    else:
        print('please specify 2 arguments', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
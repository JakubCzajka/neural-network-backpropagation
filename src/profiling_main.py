from shell import NetworkShell


if __name__ == '__main__':
    shell = NetworkShell()
    shell.onecmd('load_dataset resources/iris.data:0.7')
    shell.onecmd('create_network resources/network_structure.json')
    shell.onecmd('train mse:200:0.05:0.9:No')
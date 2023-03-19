import configs
from nim.NimWorld import NimSimWorld
from nim.nimModel import NimModel

if __name__ == "__main__":
    game = NimSimWorld(configs.size)
    model = NimModel(configs.size, '../model_dicts')

    while game.get_utility() == 0:
        print(f"Board: {game.board}")
        if game.player == 1:
            action = input("Player action:")
            try:
                game.apply(action)
            except:
                print(f"Illegal action {action}")
        else:
            actions = model.classify(game.state())
            for action in actions:
                try:
                    game.apply(action)
                    print(f"Machine played {action}")
                    break
                except:
                    pass
    if game.get_utility() == 1:
        print("Congratulations, you won!")
    else:
        print("Too bad, the machine won this time")
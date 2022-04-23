import visdom


if __name__ == '__main__':

    filenames = ['/home/chris/Documents/oodl_local/graphs/OODL_2/[tnet64|te_rot]_OODL2[32-256|r1248|p=1]__2020.12.06-14.16.56.log']

    for filename in filenames:
        vis = visdom.Visdom()
        vis.replay_log(filename)
        print("done")


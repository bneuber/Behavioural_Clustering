
def get_hexposition(row, ind, map_range):
    honeycombs = [[(row, ind)]]
    for r_ind in range(map_range):
        for h in honeycombs[r_ind]:
            if h[0] % 2 != 0:
                honeycombs_new = [(h[0]-1, h[1]-1, r_ind),
                                  (h[0]-1, h[1], r_ind),
                                  (h[0], h[1]-1, r_ind),
                                  (h[0], h[1]+1, r_ind),
                                  (h[0]+1, h[1]-1, r_ind),
                                  (h[0]+1, h[1], r_ind)]
            else:
                honeycombs_new = [(h[0]-1, h[1], r_ind),
                                  (h[0]-1, h[1]+1, r_ind),
                                  (h[0], h[1]-1, r_ind),
                                  (h[0], h[1]+1, r_ind),
                                  (h[0]+1, h[1], r_ind),
                                  (h[0]+1, h[1]+1, r_ind)]
            honeycombs += [honeycombs_new]
    honeycombs = {h for H in honeycombs for h in H}
    # mein_set = {element for unterliste in liste for element in unterliste}
    return honeycombs

        
a = get_hexposition(5, 9, 2)
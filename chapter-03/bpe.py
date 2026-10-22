def get_stats(ids):
    counts={}
    for pair in zip(ids, ids[1:]):
        counts[pair]=counts.get(pair, 0) + 1

    return counts


def merge(ids, pair, new_id):
    out, i=[], 0
    while i<len(ids):
        if i<len(ids) -1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
            out.append(new_id)
            i+=2
        else:
            out.append(ids[i])
            i+=1;

    return out



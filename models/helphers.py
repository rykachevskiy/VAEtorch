def parse_features_numbers(features_nums):
    paired_fn = []
    for a, b in zip(features_nums[:-1], features_nums[1:]):
        paired_fn.append([a,b])
    return paired_fn
def calculate_boundary_deviation(ground_truth_lists, combined_segment_lists):
    
    """
    Calculate the absolute deviation of segment boundaries for multiple combined segment lists.
    
    Args:
    ground_truth_lists (list of lists): List of ground truth segment lists
    combined_segment_lists (list of lists): List of combined segment lists
    
    Returns:
    list of dicts: Boundary deviation for start and end of each segment for each list
    """

    all_boundary_devs = []
    
    for ground_truth, combined_segments in zip(ground_truth_lists, combined_segment_lists):
        start_deviations = []
        end_deviations = []
        
        for gt_seg, comb_seg in zip(ground_truth, combined_segments):
            start_dev = abs(gt_seg[0] - comb_seg[0])
            end_dev = abs(gt_seg[1] - comb_seg[1])
            
            start_deviations.append(start_dev)
            end_deviations.append(end_dev)
        
        avg_start_dev = sum(start_deviations) / len(start_deviations)
        avg_end_dev = sum(end_deviations) / len(end_deviations)
        
        all_boundary_devs.append({
            'start_deviations': start_deviations,
            'end_deviations': end_deviations,
            'avg_start_deviation': avg_start_dev,
            'avg_end_deviation': avg_end_dev
        })
    
    return all_boundary_devs


ground_truth_list = [
    [(), (), (), ()], # Each tuple has start_frame,end_frame
]

combined_segments_list = [
    [(), (), (), ()], # Each tuple has start_frame,end_frame
]

    
print("Boundary Deviation:")
boundary_devs = calculate_boundary_deviation(ground_truth_list, combined_segments_list)
avg_start_devs = []
avg_end_devs = []
for i, boundary_dev in enumerate(boundary_devs):
    print(f"   List {i + 1} Start Deviations: {boundary_dev['start_deviations']}")
    print(f"   List {i + 1} End Deviations: {boundary_dev['end_deviations']}")
    print(f"   List {i + 1} Avg Start Deviation: {boundary_dev['avg_start_deviation']:.2f}")
    print(f"   List {i + 1} Avg End Deviation: {boundary_dev['avg_end_deviation']:.2f}")
    
    avg_start_devs.append(boundary_dev['avg_start_deviation'])
    avg_end_devs.append(boundary_dev['avg_end_deviation'])

overall_avg_start_deviation = sum(avg_start_devs) / len(avg_start_devs)
overall_avg_end_deviation = sum(avg_end_devs) / len(avg_end_devs)
print(f"   Overall Avg Start Deviation: {overall_avg_start_deviation:.2f}")
print(f"   Overall Avg End Deviation: {overall_avg_end_deviation:.2f}")
    
   






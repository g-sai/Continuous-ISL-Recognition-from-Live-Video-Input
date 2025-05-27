
def calculate_segment_matching_accuracy(ground_truth_lists, combined_segment_lists, threshold=0.5):
    
    """
    Calculate segment matching accuracy for multiple combined segment lists based on overlap threshold.
    
    Args:
    ground_truth_lists (list of lists): List of ground truth segment lists
    combined_segment_lists (list of lists): List of combined segment lists
    threshold (float): Minimum IoU to consider a segment match
    
    Returns:
    list: Matching accuracy for each segment list
    """

    matching_accuracies = []
    
    for ground_truth, combined_segments in zip(ground_truth_lists, combined_segment_lists):
        matched_segments = 0
        
        for gt_seg, comb_seg in zip(ground_truth, combined_segments):
            gt_start, gt_end = gt_seg
            comb_start, comb_end = comb_seg
            
            intersection_start = max(gt_start, comb_start)
            intersection_end = min(gt_end, comb_end)
            intersection = max(0, intersection_end - intersection_start)
            
            union_start = min(gt_start, comb_start)
            union_end = max(gt_end, comb_end)
            union = union_end - union_start
            
            iou = intersection / union if union > 0 else 0
            
            if iou >= threshold:
                matched_segments += 1
        
        matching_accuracy = (matched_segments / len(ground_truth)) * 100
        matching_accuracies.append(matching_accuracy)
    
    return matching_accuracies


ground_truth_list = [
    [(), (), (), ()], # Each tuple has start_frame,end_frame
]

combined_segments_list = [
    [(), (), (), ()], # Each tuple has start_frame,end_frame
]

print("Segment Matching Accuracy:")
matching_accuracies = calculate_segment_matching_accuracy(ground_truth_list, combined_segments_list)
for i, accuracy in enumerate(matching_accuracies):
    print(f"   List {i + 1} Matching Accuracy: {accuracy:.2f}%")
    






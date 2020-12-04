from .eval_giro import giro_evaluation

def eval_giro(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    return giro_evaluation(
                            dataset=dataset,
                            predictions=predictions,
                            box_only=box_only,
                            output_folder=output_folder,
                            iou_types=iou_types,
                            expected_results=expected_results,
                            expected_results_sigma_tol=expected_results_sigma_tol,
                             
                        )

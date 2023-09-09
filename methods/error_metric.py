# Edited from https://github.com/aymenmir1/3dpw-eval/tree/master
import numpy as np

SMPL_NR_JOINTS = 24

def sort_by_order(array, order):

    new_array = np.zeros((len(order), *array[0].shape))

    for target_idx, current_idx in enumerate(order):
        if current_idx != -1:
            new_array[target_idx] = array[current_idx]

    return new_array

def rmse(pred, gt):

    squared_errors = (pred - gt) ** 2

    mse = np.mean(squared_errors)

    rmse = np.sqrt(mse)
    
    return rmse


def match_models(preds, gts, match_by = "root"):

    distances = np.ones((len(preds), len(gts))) * float("inf")
    
    max_size = max(len(preds), len(gts))
    min_size = min(len(preds), len(gts))

    # Initialize arrays to store the matched pairs and flags to track whether an element has been matched
    matched_pairs = np.full((max_size, 2), -1, dtype=int)
    matched_pairs[:len(preds), 0] = np.arange(len(preds)) # first column is always fixed preds

    matched_flags = np.zeros(len(gts), dtype=bool)

    # calculate distance matrix
    for i, matrix1 in enumerate(preds):
        for j, matrix2 in enumerate(gts):

            if match_by == "root":
                norm = rmse(matrix1[0], matrix2[0])
            elif match_by == "joints":
                # Calculate the Frobenius norm for the difference between matrices
                norm = np.linalg.norm(matrix1 - matrix2)
            else:
                # TODO
                pass

            distances[i, j] = norm

    # match the preds with gts
    for _ in range(min_size):

        ind = np.unravel_index(np.argmin(distances, axis=None), distances.shape)

        pred_index = ind[0]
        gt_index = ind[1]

        matched_pairs[pred_index] = [pred_index, gt_index]

        distances[pred_index, :] = float("inf")
        distances[:, gt_index] = float("inf")
        matched_flags[gt_index] = True


    # if there are more gts than preds, than add the rest as unordered way
    if len(preds) < len(gts):
        unmatched_indices = np.where(~matched_flags)[0]

        for i, idx in enumerate(unmatched_indices):
            matched_pairs[len(preds) + i][1] = idx

    # if more gts than preds then match those gts with zero matrices
    if len(preds) < len(gts):
        # Sort gts based on matched indices
        sorted_gts = np.zeros_like(gts)  # Initialize with zeros
        for i, (preds_idx, gts_idx) in enumerate(matched_pairs):
            # if num_gts > num_preds, gts_idx newer be -1
            sorted_gts[i] = gts[gts_idx]

        zero_matrices = np.zeros((len(gts) - len(preds), *preds[0].shape), dtype=preds.dtype)
        appended_preds = np.append(preds, zero_matrices, axis=0)

        preds = appended_preds
        gts = sorted_gts
        
    # if more preds than gts then match those preds with zero matrices
    if len(preds) > len(gts):
        # Sort gts based on matched indices
        sorted_gts = np.zeros_like(preds)  # Initialize with zeros
        for i, (preds_idx, gts_idx) in enumerate(matched_pairs):
            if gts_idx != -1:
                sorted_gts[i] = gts[gts_idx]
            else:
                sorted_gts[i] = np.zeros((preds[0].shape), dtype=preds.dtype)

        gts = sorted_gts
    
    return preds, gts, matched_pairs


def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Ensure that the first argument is the prediction

    Source: https://en.wikipedia.org/wiki/Kabsch_algorithm

    :param S1 predicted joint positions array 24 x 3
    :param S2 ground truth joint positions array 24 x 3
    :return S1_hat: the predicted joint positions after apply similarity transform
            R : the rotation matrix computed in procrustes analysis
    '''
    # If all the values in pred3d are zero then procrustes analysis produces nan values
    # Instead we assume the mean of the GT joint positions is the transformed joint value

    if not (np.sum(np.abs(S1)) == 0):
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert (S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1 ** 2)

        # 3. The outer producshape1,t of X1 and X2.
        K = X1.dot(X2.T)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        # Construct R.
        R = V.dot(Z.dot(U.T))

        # 5. Recover scale.
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale * (R.dot(mu1))

        # 7. Error:
        S1_hat = scale * R.dot(S1) + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat, R
    else:
        S1_hat = np.tile(np.mean(S2, axis=0), (SMPL_NR_JOINTS, 1))
        R = np.identity(3)

        return S1_hat, R

def align_by_root(joints):
    """
    Assumes joints is 24 x 3 in SMPL order.
    Subtracts the location of the root joint from all the other joints
    """
    root = joints[0, :]

    return joints - root


def compute_mpjpe(preds3d, gt3ds, mask=None):
    """
    Gets MPJPE after root alignment + MPJPE after Procrustes.
    Evaluates on all the 24 joints joints.
    Inputs:
    :param gt3ds: N x 24 x 3
    :param preds: N x 24 x 3
    :returns
        MPJPE : scalar - mean of all MPJPE errors
        MPJPE_PA : scalar- mean of all MPJPE_PA errors
    """

    errors, errors_pa = [], []

    for i, (gt3d, pred3d) in enumerate(zip(gt3ds, preds3d)):
        # gt3d = gt3d.reshape(-1, 3)
        # Root align.
        gt3d = align_by_root(gt3d)
        pred3d = align_by_root(pred3d)

        # Compute MPJPE_PA and also store similiarity matrices to apply them later to rotation matrices for MPJAE_PA
        pred3d_sym, R = compute_similarity_transform(pred3d, gt3d)

        if mask is not None:
            gt3d = gt3d[mask[i], :]
            pred3d = pred3d[mask[i], :]
            pred3d_sym = pred3d_sym[mask[i], :]

        # Compute MPJPE
        joint_error = np.sqrt(np.sum((gt3d - pred3d) ** 2, axis=1))
        errors.append(np.mean(joint_error))

        # Compute MPJPE_PA and also store similiarity matrices to apply them later to rotation matrices for MPJAE_PA
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
        errors_pa.append(np.mean(pa_error))


    return np.mean(np.array(errors)), np.mean(np.array(errors_pa))

def compute_shape_error(preds, groundtruths): # use maybe rmse for shape errors 
    return rmse(preds, groundtruths)


if __name__ == "__main__":

    num_preds = 1
    num_gts = 2

    thetas_gt = np.random.rand(num_gts, 24, 3)
    thetas_pred = np.random.rand(num_preds, 24, 3)

    betas_gt = np.random.rand(num_gts, 10)
    betas_pred = np.random.rand(num_preds, 10)

    occlusion_mask = np.random.choice([True, False], size=(num_gts, 24))

    # align if num preds and num gts do not match

    thetas_pred, thetas_gt, matched_pairs = match_models(preds=thetas_pred, gts=thetas_gt, match_by="root")
    
    betas_pred = sort_by_order(array=betas_pred, order=matched_pairs[:,0])
    betas_gt = sort_by_order(array=betas_gt, order=matched_pairs[:,1])

    mpjpe, pa_mpjpe = compute_mpjpe(thetas_pred, thetas_gt, mask=occlusion_mask)

    shape_error = compute_shape_error(betas_pred, betas_gt)

    pose_coeff = 0.9
    shape_coeff = 0.1

    total_error = pose_coeff*mpjpe + shape_coeff*shape_error

    print(total_error)


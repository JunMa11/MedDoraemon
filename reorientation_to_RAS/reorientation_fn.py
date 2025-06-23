import nibabel as nib
import nibabel.orientations as nio
import numpy as np
import os
import SimpleITK as sitk

def reorient_to_true_RAS_for_itksnap(input_path, output_path=None):
    """
    Fully reorients image to RAS (Right-Anterior-Superior) using nibabel,
    then sets spacing, origin, and direction into a SimpleITK image for ITK-SNAP compatibility.
    """
    # --- Load image with nibabel ---
    nii = nib.load(input_path)
    orig_affine = nii.affine
    orig_data = nii.get_fdata()
    orig_axcodes = nio.aff2axcodes(orig_affine)
    desired_axcodes = ('R', 'A', 'S')

    print(f"Original orientation: {orig_axcodes}")

    # --- Reorient to RAS if needed ---
    if orig_axcodes != desired_axcodes:
        print(f"Reorienting from {orig_axcodes} to {desired_axcodes}")
        orig_ornt = nio.axcodes2ornt(orig_axcodes)
        desired_ornt = nio.axcodes2ornt(desired_axcodes)
        transform = nio.ornt_transform(orig_ornt, desired_ornt)
        reoriented_nifti = nii.as_reoriented(transform)
        reoriented_data = reoriented_nifti.get_fdata()
        new_affine = reoriented_nifti.affine
    else:
        print("Image already in RAS orientation.")
        reoriented_data = orig_data
        new_affine = orig_affine

    # --- Convert data to sitk format (Z, Y, X) ---
    data_for_sitk = np.transpose(reoriented_data, (2, 1, 0))  # [Z, Y, X]
    sitk_img = sitk.GetImageFromArray(data_for_sitk)

    # --- Compute spacing, origin, direction from affine ---
    RZS = new_affine[:3, :3]
    spacing = np.linalg.norm(RZS, axis=0)
    direction_matrix = RZS / spacing
    direction_flat = direction_matrix.flatten(order='F')  # ITK expects column-major

    origin = new_affine[:3, 3]

    # --- Set metadata ---
    sitk_img.SetSpacing(tuple(spacing))
    sitk_img.SetOrigin(tuple(origin))
    sitk_img.SetDirection(tuple(direction_flat))

    # --- Save output ---
    if output_path is None:
        base, _ = os.path.splitext(os.path.splitext(input_path)[0])
        output_path = f"{base}_RAS_ITKSNAP.nii.gz"

    sitk.WriteImage(sitk_img, output_path)
    print(f"✅ Saved image with RAS orientation to: {output_path}")

    # --- Return image + metadata for confirmation ---
    props = {
        'sitk_stuff': {
            'spacing': sitk_img.GetSpacing(),
            'origin': sitk_img.GetOrigin(),
            'direction': sitk_img.GetDirection()
        }
    }
    image_np = sitk.GetArrayFromImage(sitk_img)[np.newaxis]  # [1, Z, Y, X]
    return image_np, props

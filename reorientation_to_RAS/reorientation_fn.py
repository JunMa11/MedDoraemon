import nibabel as nib
import nibabel.orientations as nio
import numpy as np
import os
import SimpleITK as sitk


def set_sitk_metadata_from_affine(sitk_img, affine):
    spacing = tuple(np.linalg.norm(affine[:3, i]) for i in range(3))
    origin = tuple(float(affine[i, 3]) for i in range(3))
    direction = tuple(float(x) for x in affine[:3, :3].flatten(order='F'))

    sitk_img.SetSpacing(spacing)
    sitk_img.SetOrigin(origin)
    sitk_img.SetDirection(direction)
    return sitk_img


def reorient_to_true_RAS_for_itksnap(input_path, output_path=None):
    """
    Fully reorients image to RAS (Right-Anterior-Superior) for ITK-SNAP.
    Uses identity direction and reoriented voxel order to ensure display shows RAS.
    """
    nii = nib.load(input_path)
    orig_affine = nii.affine
    orig_data = nii.get_fdata()
    orig_axcodes = nio.aff2axcodes(orig_affine)
    desired_axcodes = ('L', 'P', 'I') # DO NOT change this if you want the orientation to be RAS for ITK-SNAP

    print(f"Original orientation: {orig_axcodes}")

    if orig_axcodes != desired_axcodes:
        print(f"Reorienting from {orig_axcodes} to {desired_axcodes}")
        orig_ornt = nio.axcodes2ornt(orig_axcodes)
        desired_ornt = nio.axcodes2ornt(desired_axcodes)
        transform = nio.ornt_transform(orig_ornt, desired_ornt)
        reoriented_data = nio.apply_orientation(orig_data, transform)
    else:
        print("Image already in RAS orientation.")
        reoriented_data = orig_data

    # Convert to SimpleITK format
    data_for_sitk = np.transpose(reoriented_data, (2, 1, 0))  # [Z, Y, X]
    sitk_img = sitk.GetImageFromArray(data_for_sitk)

    # Force identity direction matrix (RAS)
    sitk_img.SetDirection(np.eye(3).flatten())  # [1, 0, 0, 0, 1, 0, 0, 0, 1]

    # Optionally compute spacing and origin from affine (use only if affine is meaningful)
    spacing = tuple(np.linalg.norm(orig_affine[:3, i]) for i in range(3))
    origin = tuple(float(orig_affine[i, 3]) for i in range(3))

    sitk_img.SetSpacing(spacing)
    sitk_img.SetOrigin(origin)

    if output_path is None:
        base, ext = os.path.splitext(os.path.splitext(input_path)[0])
        output_path = f"{base}_RAS_ITKSNAP.nii.gz"

    sitk.WriteImage(sitk_img, output_path)
    print(f"✅ Saved image with forced RAS orientation to: {output_path}")

    # Return image + metadata
    props = {
        'sitk_stuff': {
            'spacing': sitk_img.GetSpacing(),
            'origin': sitk_img.GetOrigin(),
            'direction': sitk_img.GetDirection()
        }
    }
    image_np = sitk.GetArrayFromImage(sitk_img)[np.newaxis]  # [1, Z, Y, X]
    return image_np, props

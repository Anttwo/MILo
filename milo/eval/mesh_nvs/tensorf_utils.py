import torch
import torch.nn as nn
import numpy as np

# From https://github.com/apchenstu/TensoRF/ ####
def positional_encoding(positions, freqs):
    
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def eval_sh_bases(deg, dirs):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param deg: int SH max degree. Currently, 0-4 supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., (deg+1) ** 2)
    """
    assert deg <= 4 and deg >= 0
    result = torch.empty((*dirs.shape[:-1], (deg + 1) ** 2), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = C0
    if deg > 0:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -C1 * y;
        result[..., 2] = C1 * z;
        result[..., 3] = -C1 * x;
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = C2[0] * xy;
            result[..., 5] = C2[1] * yz;
            result[..., 6] = C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = C2[3] * xz;
            result[..., 8] = C2[4] * (xx - yy);

            if deg > 2:
                result[..., 9] = C3[0] * y * (3 * xx - yy);
                result[..., 10] = C3[1] * xy * z;
                result[..., 11] = C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = C3[5] * z * (xx - yy);
                result[..., 15] = C3[6] * x * (xx - 3 * yy);

                if deg > 3:
                    result[..., 16] = C4[0] * xy * (xx - yy);
                    result[..., 17] = C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result

def SHRender(xyz_sampled, viewdirs, features, sh_degree=2):
    sh_mult = eval_sh_bases(sh_degree, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb

# def SHRender(xyz_sampled, viewdirs, features):
#     sh_mult = eval_sh_bases(2, viewdirs)[:, None]
#     rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
#     rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
#     return rgb

class MLPRender_Fea_Diffuse(torch.nn.Module):
    def __init__(self,inChanel, feape=6, featureC=128, output_dim=3):
        super(MLPRender_Fea_Diffuse, self).__init__()

        self.in_mlpC = 2*feape*inChanel + inChanel
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, output_dim)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, features):
        indata = [features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)

        return rgb
    
# From RefNeRF: https://github.com/google-research/multinerf/blob/main/internal/ref_utils.py#L84
def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    # Convert list into a numpy array.
    ml_array = np.array(ml_list).T
    return ml_array

def generalized_binomial_coeff(a, k):
  """Compute generalized binomial coefficients."""
  return np.prod(a - np.arange(k)) / np.math.factorial(k)



def assoc_legendre_coeff(l, m, k):
  """Compute associated Legendre polynomial coefficients.

  Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
  (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

  Args:
    l: associated Legendre polynomial degree.
    m: associated Legendre polynomial order.
    k: power of cos(theta).

  Returns:
    A float, the coefficient of the term corresponding to the inputs.
  """
  return ((-1)**m * 2**l * np.math.factorial(l) / np.math.factorial(k) /
          np.math.factorial(l - k - m) *
          generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))


def sph_harm_coeff(l, m, k):
  """Compute spherical harmonic coefficients."""
  return (np.sqrt(
      (2.0 * l + 1.0) * np.math.factorial(l - m) /
      (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))

    
def generate_ide_fn(deg_view):
  """Generate integrated directional encoding (IDE) function.

  This function returns a function that computes the integrated directional
  encoding from Equations 6-8 of arxiv.org/abs/2112.03907.

  Args:
    deg_view: number of spherical harmonics degrees to use.

  Returns:
    A function for evaluating integrated directional encoding.

  Raises:
    ValueError: if deg_view is larger than 5.
  """
  if deg_view > 5:
    raise ValueError('Only deg_view of at most 5 is numerically stable.')

  ml_array = get_ml_array(deg_view)
  l_max = 2**(deg_view - 1)

  # Create a matrix corresponding to ml_array holding all coefficients, which,
  # when multiplied (from the right) by the z coordinate Vandermonde matrix,
  # results in the z component of the encoding.
  mat = np.zeros((l_max + 1, ml_array.shape[1]))
  for i, (m, l) in enumerate(ml_array.T):
    for k in range(l - m + 1):
      mat[k, i] = sph_harm_coeff(l, m, k)

  def integrated_dir_enc_fn(xyz, kappa_inv):
    """Function returning integrated directional encoding (IDE).

    Args:
      xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
      kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
        Mises-Fisher distribution.

    Returns:
      An array with the resulting IDE.
    """
    x = xyz[..., 0:1]
    y = xyz[..., 1:2]
    z = xyz[..., 2:3]

    # Compute z Vandermonde matrix.
    vmz = torch.cat([z**i for i in range(mat.shape[0])], dim=-1)

    # Compute x+iy Vandermonde matrix
    vmxy = torch.cat([(x + 1j * y)**m for m in ml_array[0, :]], dim=-1)

    # Get spherical harmonics.
    sph_harms = vmxy * torch.matmul(vmz, torch.from_numpy(mat).float().to(xyz.device))

    # Apply attenuation function using the von Mises-Fisher distribution
    # concentration parameter, kappa.
    sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)
    sigma = torch.from_numpy(sigma).float().to(xyz.device)
    ide = sph_harms * torch.exp(-sigma * kappa_inv)

    # Split into real and imaginary parts and return
    return torch.cat([torch.real(ide), torch.imag(ide)], dim=-1)
  return integrated_dir_enc_fn

#####################
    
class MLPRender_Fea_Specular(torch.nn.Module):
    def __init__(self,inChanel, feape=6, featureC=128, output_dim=3):
        super(MLPRender_Fea_Specular, self).__init__()

        self.integrated_pos_enc = generate_ide_fn(deg_view=5)
        self.integrated_pos_enc_dim = self.integrated_pos_enc(torch.randn(1, 3), torch.randn(1, 1)).shape[-1]
        self.in_mlpC = 2*feape*inChanel + inChanel + self.integrated_pos_enc_dim + 1
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, output_dim)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, features, omega_r, rho, omega_o_dot_normal):
        indata = [features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]

        # Apply IDE from RefNeRF
        direction_positional_encoding = self.integrated_pos_enc(omega_r, rho)

        indata += [direction_positional_encoding] # (..., D)
        indata += [omega_o_dot_normal] # (..., 1)

        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)

        return rgb

class MLPRender_Fea(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        count_w = max(count_w, 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
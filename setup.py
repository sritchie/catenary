from setuptools import setup, find_packages

# This follows the style of Jaxlib installation here:
# https://github.com/google/jax#pip-installation
PYTHON_VERSION = "cp36"
CUDA_VERSION = "cuda100"  # alternatives: cuda90, cuda92, cuda100, cuda101
PLATFORM = "linux_x86_64"  # alternatives: linux_x86_64
BASE_URL = "https://storage.googleapis.com/jax-releases"


def jax_artifact(version, gpu=False):
  if gpu:
    prefix = f"{BASE_URL}/{CUDA_VERSION}/jaxlib"
    wheel_suffix = f"{PYTHON_VERSION}-none-{PLATFORM}.whl"
    location = f"{prefix}-{version}-{wheel_suffix}"
    return f"jaxlib @ {location}"

  return "jaxlib"


with open('README.md') as rf:
  readme = rf.read()


def with_versioneer(f, default=None):
  """Attempts to execute the supplied single-arg function by passing it
versioneer if available; else, returns the default.

  """
  try:
    import versioneer
    return f(versioneer)
  except ModuleNotFoundError:
    return default


JAX_VERSION = "0.1.39"
REQUIRED_PACKAGES = ["numpy>=1.18.0", "tqdm>=4.42.1", "fs", "fs-gcsfs", "jax"]

setup(
    name='catenary',
    version=with_versioneer(lambda v: v.get_version()),
    cmdclass=with_versioneer(lambda v: v.get_cmdclass(), {}),
    description='Hunting for new string theories.',
    long_description=readme,
    author='Blueshift Team',
    author_email='samritchie@google.com',
    url='https://team.git.corp.google.com/blueshift/catenary',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        "cpu": [jax_artifact(JAX_VERSION, gpu=False)],
        "gpu": [jax_artifact(JAX_VERSION, gpu=True)],
    },
    include_package_data=True,
)

{
  description = "DEQ devshell";

  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    };

    # General packages.
    python-packages = ps:
      with ps; [
        pip
        setuptools
        virtualenv
      ];
    packages = [
      (pkgs.python313.withPackages python-packages)
      pkgs.just
      pkgs.kaggle
      pkgs.uv
    ];

    # Necessary libraries.
    libs = [
      pkgs.cudaPackages.cudatoolkit
      pkgs.cudaPackages.cudnn
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib

      # Where your local "lib/libcuda.so" lives. If you're not on NixOS, you should
      # provide the right path (likely another one).
      "/run/opengl-driver"
    ];
  in {
    devShells.${system}.default = pkgs.mkShell {
      name = "deq";
      inherit packages;

      env = {
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
        # Tells jax where to find libdevice.10.bc.
        # https://github.com/jax-ml/jax/discussions/6479#discussioncomment-622839
        XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudatoolkit}";
      };
    };
  };
}

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

    python-packages = ps:
      with ps; [
        pip
        setuptools
        virtualenv
      ];

    fhs = pkgs.buildFHSUserEnv {
      name = "deq";
      targetPkgs = pkgs: [
        (pkgs.python313.withPackages python-packages)
        pkgs.cudaPackages.cudatoolkit
        pkgs.just
        pkgs.kaggle
        pkgs.uv
      ];
      multiPkgs = pkgs: [
        pkgs.zlib # Numpy dep.
      ];
      profile = ''
        export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
      '';
    };
  in {
    devShells.${system}.default = fhs.env;
  };
}

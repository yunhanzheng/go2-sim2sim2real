# Empty stub for torchvision.
#
# rsl_rl/modules/depth_backbone.py does `import torchvision` at module load, and
# actor_critic.py imports depth_backbone — but the deploy inference path
# (Actor + Estimator, depth latent supplied externally) never calls into the depth
# backbone. torchvision is not installed in the ros_humble env (and its ABI mismatches
# this torch build), so this empty module satisfies the import without pulling in the
# real package. Mirrors the in-process stub in scripts/distill_runner.py.
#
# Put deploy_mac/stubs first on PYTHONPATH so `import torchvision` resolves here.

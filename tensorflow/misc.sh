# in case you run into the following error with tensorboard
#
# ValueError: Duplicate plugins for name projector
#
# The solution that worked was from here : https://github.com/pytorch/pytorch/issues/22676
# basically uninstall all things TF and then re-install it
#
pip3 uninstall tb-nightly tensorboard tensorflow-estimator tensorflow-gpu tf-estimator-nightly
pip3 install /tmp/tensorflow_pkg/tensorflow-2.0.0-cp35-cp35m-linux_x86_64.whl 

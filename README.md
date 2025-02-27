Below is an overview of how model creation works using openism_model_creator, what general steps have been taken to create the model, and what has been done so far in terms of muscle creation.

Bone Creation: 

The primary inputs are participant specific tracking data (.trc files) related to a static (static.trc) and dynamic (kneeoptimisation.trc, along with a static pickle file (static.pkl, unsure what this is FYI).

Model creation begins by utilising an articulated shape model (asm) to generate meshes that better represent the underlying bone geometry of the participant, as part of the asm's workflow it also predicts anatomical landmarks which are retained.

As the asm does not generate foot specific meshes, the gait 2392 feet meshes are initially loaded to be repurposed and fit to each participant.

The first body to be created is the pelvis, a realignment of the pelvis is performed using the ASISs. This rotates the pelvis so that the vector between the ASISs is aligned with the horizontal axis (i.e side to side/ left to right etc), this was done as participants may be standing slightly rotated during their static trial, where the generated meshes (by asm) retain this slight rotation. This ensures that the pelvis rotation variable tracked within Opensim is consistent between participants where the "0" position is the pelvis facing perfectly forward. No reorientations were applied to posterior/ anterior tilt, or pelvic obliquity. Markers from the static trial and anatomical landmarks are then added to the pelvis body/mesh. 

Secondly the Femur bodies are created, and the markers/anatomical landmarks are attached to the meshes. As participants have pathologies that make it difficult to stand in a commonly neutral pose, the femurs (and other limbs) have the option to be reorientated to a more neutral pose, this was done to ensure accurate representation of joint angles. Reorientation of the femurs currently places the femoral epicondyles in a horizontal position (as above with the ASISs for the pelvis), as well as positioning the epicondylar midpoint vertically beneath the hip joint centre. The hip joint orientation was defined to as best match ISB coordinate standards with the flexion extension axis aligned with the ASISs and the rotational axis aligned with a vector passing form the hip joint centre to the epicondylar midpoint, with the remaining axis (adduction/abduction) being the cross product between the 2 aforementioned axes. As joint axes need to be orthogonal to one another, the flexion/extension axis was chosen as the primary axis and perfectly fits to that of its anatomical definition (ASISs), whilst the rotational axis (hip joint centre to epicondylar midpoint) was numerically optimised to be as close as possible whilst keeping the flexion/extension axis in place, the remaining vector as stated was the remaining orthogonal axis.


Thirdly the tibfib bodies were created, along with attaching the markers/anatomical landmarks to the meshes. As above, reorientation from the participants initial pose is possible. The tibfib's initial flexion/extension position was adjusted to position the medial malleoli beneath the medial epicondyle (essentially having the femur and tibfib be vertically aligned for the '0' degrees of flexion/extension for both the hip and knee joints). The knee joint (1 dof, pin joint) went through 2 stages, initially the flexion/extension axis was defined as the vector between the epicondyles, before being numerically optimised through using a walking trial and perturbing the orientation of this joint axis to see if any reductions in marker RMS error are observed. 

Fourthly, the initial feet bodies from gait 2392 are attached to the model and repositioned to have the talus sitting in the centre of the malleoli, the flexion extension vector for the ankle is also defined using the vector between the malleoli. The subtalar joint angle is maintained as is present in the gait 2392 model, this update is hard coded within the next section on general model updates. From here the feet bodies are reorientated to take into account patient specific foot positioning (as gait 2392 has the feet aligned perfectly in a forward/backward direction, but for participants this is not always the case). This rotates the feet bodies about the y and z axis to align the vector between the toe and heel model markers with that of the toe and heel static trial marker data. In addition the feet are rotated to a position that is considered to be flat with the ground when the joint angles are at their '0' positions.

An additional process that occurs after the feet are finalised (but are un-scaled), is a general overall model update where the joint coordinate names and ranges are updated, body segment parameters (mass and inertias) are computed based off the height and weight of the participant(using information and equations provided by Winter, 2009, as mentioned above the subtalar joint is updated, also we appropriately set the mesh paths of the feet to the correct location within the higher_level_inputs folder.


Following this process, the feet are then scaled by their experimental static trial data (other limbs are not, as this is taken into account by the asm).

Knee joint optimisation, described above, occurs next.

Following this, we moved the model markers to reduce RMS error, this is done via determining the mean error vector between the model markers and the experimental data and then applying that offset to reduce RMS error.

Validation has not yet been conducted on models produced using this tool.

Muscle Creation: 


Muscle creation is an in-the-works optional capability of opensim_model_creator, it currently has the capacity to generate muscles between the pelvis, femur, tibia and fibula with appropriate insertion and origin location, it has the capacity to break a muscle (such as the glut med) into multiple "fibers" using principal component analysis to distribute the fibers across the overall muscle attachment surface. The muscles exist with generic muscle parameters and without any form of muscle wrapping. The sacrum and spine currently do not exist within the model at all and as such muscles originating from these do not exist (glut max exists, just doesn't have the ability to have origins on the sacrum as it is not present within the model). Origins and insertions for muscles of the feet currently are not present.


To add a new participant, create a new folder with the participants name, create a folder called "Inputs", in there place the required files (listed previously) - this is for running test cases, implementation with Tim & Laura's tools follows a different method of use.

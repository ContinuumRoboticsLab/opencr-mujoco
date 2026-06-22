# Wrench Application Methods for TDCR Evaluation

## Current Issue
When applying wrenches (force + torque) to different bodies in the TDCR, the effective moment changes because:
- MuJoCo applies forces at the body's center of mass (COM)
- MuJoCo applies torques about the body's COM
- Different bodies have different COM locations, creating different moment arms

## Solution Options

### Option 1: Use Sites for Consistent Force Application Points
Add sites to your TDCR model at specific locations where forces should be applied:

```xml
<!-- In your TDCR model -->
<worldbody>
    <body name="link_25">
        <!-- Existing body definition -->
        <site name="force_site_link_25" pos="0 0 0.05" size="0.001"/>
    </body>
    <body name="EE_pos">
        <!-- Existing body definition -->
        <site name="force_site_tip" pos="0 0 0" size="0.001"/>
    </body>
</worldbody>
```

Then modify the evaluator to apply forces at sites:

```python
def apply_wrench_at_site(model, data, site_id, force, torque):
    """Apply a wrench at a specific site location."""
    # Get site position and body
    site_pos = data.site(site_id).xpos
    body_id = model.site_bodyid[site_id]
    # IMPORTANT: xfrc_applied acts at the body's center of mass (xipos),
    # not its frame origin (xpos) — the moment arm must use the COM.
    body_com = data.xipos[body_id]

    # Compute moment from force applied at site about body COM
    r = site_pos - body_com
    moment_from_force = np.cross(r, force)

    # Total torque = original torque + moment from offset force
    total_torque = torque + moment_from_force

    # Apply to body
    data.xfrc_applied[body_id][:3] = force
    data.xfrc_applied[body_id][3:] = total_torque
```

### Option 2: Use Generalized Forces (qfrc_applied)
Convert spatial forces to generalized forces using Jacobians:

```python
def apply_spatial_force_at_point(model, data, body_id, point, force, torque):
    """Apply force at a specific point in space."""
    # Allocate jacobians
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    
    # Get jacobian for the body
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    
    # Compute moment about the point
    body_pos = data.body(body_id).xpos
    r = point - body_pos
    total_torque = torque + np.cross(r, force)
    
    # Convert to generalized forces
    qfrc = jacp.T @ force + jacr.T @ total_torque
    data.qfrc_applied[:] += qfrc
```

### Option 3: Standardize Reference Point
Apply all wrenches about a common reference point (e.g., world origin):

```python
def apply_wrench_about_origin(model, data, body_id, force, torque_about_origin):
    """Apply wrench specified about world origin."""
    # Get body position
    body_pos = data.body(body_id).xpos
    
    # Convert torque from origin to body COM
    moment_from_force = np.cross(body_pos, force)
    torque_about_com = torque_about_origin - moment_from_force
    
    # Apply to body
    data.xfrc_applied[body_id][:3] = force
    data.xfrc_applied[body_id][3:] = torque_about_com
```

### Option 4: Pure Forces Only (Simplest)
If the torque component isn't critical, apply only forces:

```python
# Only use first 3 components of wrench
data.xfrc_applied[mid_body_id][:3] = mid_wrench[:3]
data.xfrc_applied[tip_body_id][:3] = tip_wrench[:3]
```

## Recommendation
For most accurate replication of reference data:
1. Determine how forces were applied in the reference (SoroSim) simulation
2. If forces were applied at specific geometric points, use Option 1 (sites)
3. If the reference used a different convention, implement the matching approach
4. Document the chosen method clearly for reproducibility

The sites approach (Option 1) is recommended as it:
- Provides explicit, visible force application points
- Is easier to debug and visualize
- Maintains consistency across different body geometries
- Matches common experimental setups where forces are applied at specific locations
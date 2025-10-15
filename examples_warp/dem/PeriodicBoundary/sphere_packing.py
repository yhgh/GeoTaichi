from geowarp import *

init()

dem = DEM()

dem.set_configuration(domain=[20.,25.,35.],
                      boundary=["Period", "Period", "Destroy"],
                      gravity=[0.,0.,-9.8],
                      engine="SymplecticEuler",
                      search="LinkedCell")

dem.set_solver({
                "Timestep":         1e-4,
                "SimulationTime":   15,
                "SaveInterval":     0.3,
                "SavePath":         "OutputData"
               })

dem.memory_allocate(memory={
                            "max_material_number": 2,
                            "max_particle_number": 100000,
                            "max_sphere_number": 100000,
                            "max_clump_number": 0,
                            "max_plane_number": 2,
                            "verlet_distance_multiplier": 0.1,
                            "body_coordination_number": 16,
                            "wall_coordination_number": 3,
                            "compaction_ratio": [0.15, 0.05]
                            }, log=True)                       

dem.add_attribute(materialID=0,
                  attribute={
                            "Density":            2650,
                            "ForceLocalDamping":  0.,
                            "TorqueLocalDamping": 0.
                            })
                            
dem.add_attribute(materialID=1,
                  attribute={
                            "Density":            26500,
                            "ForceLocalDamping":  0.,
                            "TorqueLocalDamping": 0.
                            })

dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "GroupID": 0,
                               "MaterialID": 0,
                               "File":'SpherePacking.txt',
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"]
                               }}) 

dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                         particle_wall_contact_model="Hertz Mindlin Model")
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "ShearModulus":               4.3e6,
                            "Poisson":                    0.3,
                            "Friction":                   0.5,
                            "Restitution":                0.6
                           })
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "ShearModulus":               7.9e6,
                            "Poisson":                    0.3,
                            "Friction":                   0.5,
                            "Restitution":                0.6
                           })
                    
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([20, 7.5, 0.]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([20, 7.5, 35]),
                   "OuterNormal":  ti.Vector([0., 0., -1.])
                  })

dem.select_save_data()

dem.run()            

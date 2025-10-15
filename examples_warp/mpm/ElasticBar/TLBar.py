from geowarp import *

init()

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([5., 3., 5.]), 
                      background_damping=0.0, 
                      alphaPIC=0.01, 
                      mapping="MUSL", 
                      stabilize=None,
                      configuration="TLMPM")

mpm.set_solver(solver={
                           "Timestep":                   1e-4,
                           "SimulationTime":             6,
                           "SaveInterval":               0.05
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    8400,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   80935
                                                          }
                            })

mpm.add_material(model="NeoHookean",
                 material={
                               "MaterialID":           1,
                               "Density":              800.,
                               "YoungModulus":         8e6,
                               "PoissonRatio":         0.3
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.1, 0.1, 0.1])
                        })

mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.5, 1., 3.5]),
                            "BoundingBoxSize": ti.Vector([4, 0.5, 0.5]),
                            
                      })

mpm.add_body(body={
                       "Template": {
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }
                   })

mpm.add_boundary_condition(boundary=[
                                        {    
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0., 0.],
                                             "StartPoint":     [0.5, 0, 0],
                                             "EndPoint":       [0.5, 3., 5.]
                                        }
                                    ])

mpm.select_save_data()

mpm.run()

mpm.postprocessing()

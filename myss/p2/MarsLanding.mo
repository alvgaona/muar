model MarsLanding
  parameter Real force1 = 36350;     // Initial deceleration
  parameter Real force2 = 3605;      // Terminal descent thrust
  protected
    parameter Real thrustEndTime = 210;
    parameter Real thrustDecayTime = 43.2;
  public
    Rocket starship(name="Starship");  // New spaceship
    CelestialBody mars(
      name="Mars",
      mass=6.4171e23,
      radius=3.3895e6
    );
equation
  // Altitude and velocity-based thrust control
  starship.thrust = if (time < thrustDecayTime) then force1
                  else if (time < thrustEndTime) then force2
                  else 0;
  starship.gravity = mars.g*mars.mass/(starship.altitude+mars.radius)^2;
end MarsLanding;

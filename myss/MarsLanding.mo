model MarsLanding
  parameter Real force1 = 43000;     // Initial deceleration
  parameter Real force2 = 8000;      // Increased mid-phase thrust
  parameter Real force3 = 15000;     // Terminal descent thrust
  protected
    parameter Real phase1Time = 50;
    parameter Real phase2Time = 180;
    // Remove hard cutoff, use altitude-based control
  public
    Rocket starship(name="Starship", altitude(start=100000));
    CelestialBody mars(name="Mars",mass=6.4171e23,radius=3.3895e6);
equation
  // Altitude and velocity-based thrust control
  starship.thrust = if (time < phase1Time) then force1
                  else if (starship.altitude > 5000) then force2
                  else if (starship.altitude > 100 and abs(starship.velocity) > 10) then force3
                  else if (starship.altitude > 0) then
                    min(force3, max(1000, abs(starship.velocity) * starship.mass * 2))
                  else 0;
  starship.gravity = mars.g*mars.mass/(starship.altitude+mars.radius)^2;
end MarsLanding;

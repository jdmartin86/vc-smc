import tensorflow as tf

def angle_difference(a,b):
    """
    Compute the element-wise difference between `a` and `b` on the SO(1) manifold

    Args:
      a: Tensor of angles (not necessarily normalized) (source angle)
      b: Same type as a (target angles)

    Returns:
      diffs: Tensor with same shape as a and b containing element-wise angle
    differences between `a` and `b`, normalized to be in range [-pi,pi]

    """
    diffs = tf.atan2(tf.cos(b - a), tf.sin(b - a))
    return diffs

if __name__ == '__main__':
    """
    Unit tests for metrics
    """
    predicted_angles = tf.linspace(-10.0, 10.0, 10)
    true_angles = tf.linspace(10.0, 20.0, 10)
    differences = angle_difference(predicted_angles, true_angles)
    sess = tf.Session()
    differences = sess.run([differences])
    print(differences)

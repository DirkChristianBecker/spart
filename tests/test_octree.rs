#[path = "shared.rs"]
mod shared;
use shared::*;

use spart::{geometry::Point3D, octree::Octree};
use tracing::{debug, info};

#[test]
fn octree_3d_test() {
    info!("Starting Octree 3D test");

    // Create an octree with the shared cube boundary and capacity.
    let boundary = BOUNDARY_CUBE;
    let mut tree = Octree::new(&boundary, CAPACITY);
    info!("Created octree with boundary: {:?}", boundary);

    // Insert common 3D points into the octree.
    let points = common_points_3d();
    for pt in &points {
        tree.insert(pt.clone());
        debug!("Inserted 3D point: {:?}", pt);
    }
    info!("Finished inserting {} points", points.len());

    // Perform a kNN search.
    let target = target_point_3d();
    info!("Performing 3D kNN search for target: {:?}", target);
    let knn_results = tree.knn_search(&target, KNN_COUNT);
    info!("3D kNN search returned {} results", knn_results.len());
    assert_eq!(
        knn_results.len(),
        KNN_COUNT,
        "Expected {} nearest neighbors in 3D, got {}",
        KNN_COUNT,
        knn_results.len()
    );
    let mut prev_dist = 0.0;
    for pt in &knn_results {
        let d = distance_3d(&target, pt);
        debug!("3D kNN: Point {:?} at distance {}", pt, d);
        assert!(
            d >= prev_dist,
            "3D kNN results not sorted by increasing distance"
        );
        prev_dist = d;
    }

    // Perform a range search using a query point and radius.
    let range_query = range_query_point_3d();
    info!(
        "Performing 3D range search for query point {:?} with radius {}",
        range_query, RADIUS
    );
    let range_results = tree.range_search(&range_query, RADIUS);
    info!("3D range search returned {} points", range_results.len());
    for pt in &range_results {
        let d = distance_3d(&range_query, pt);
        debug!("3D Range: Point {:?} at distance {}", pt, d);
        assert!(
            d <= RADIUS,
            "Point {:?} is at distance {} exceeding radius {}",
            pt,
            d,
            RADIUS
        );
    }
    assert!(
        range_results.len() >= 5,
        "Expected at least 5 points in 3D range search, got {}",
        range_results.len()
    );

    info!("Octree 3D test completed successfully");
}

fn do_node_test(c : Option<&Octree<&str>>, name : Option<&str>, x : f64, y : f64, z : f64) {
    assert!(c.is_some());
    assert_eq!(c.unwrap().child_count(), 0);
    assert_eq!(c.unwrap().max_positions(), 1);

    let pos = c.unwrap().position(0);
    assert_eq!(pos.data, name);
    assert_eq!(c.unwrap().position(0).x, x);
    assert_eq!(c.unwrap().position(0).y, y);
    assert_eq!(c.unwrap().position(0).z, z);
}

#[test]
fn test_octree_2() {
    // Create an octree with the shared cube boundary and capacity.
    let boundary = BOUNDARY_CUBE;
    let mut tree = Octree::new(&boundary, 1);
    info!("Created octree with boundary: {:?}", boundary);

    tree.insert(Point3D::new( 0.0,  0.0, 51.0, Some("A")));
    tree.insert(Point3D::new( 0.0, 51.0, 51.0, Some("B")));
    tree.insert(Point3D::new( 0.0,  0.0,  0.0, Some("C")));
    tree.insert(Point3D::new( 0.0, 51.0,  0.0, Some("D")));
    tree.insert(Point3D::new(51.0,  0.0, 51.0, Some("E")));
    tree.insert(Point3D::new(51.0, 51.0, 51.0, Some("F")));
    tree.insert(Point3D::new(51.0,  0.0,  0.0, Some("G")));
    tree.insert(Point3D::new(51.0, 51.0,  0.0, Some("H")));

    assert_eq!(tree.child_count(), 8);
    assert_eq!(tree.max_positions(), 0);

    do_node_test(tree.child(0), Some("C"),  0.0, 0.0,  0.0);
    do_node_test(tree.child(1), Some("G"), 51.0, 0.0,  0.0);
    do_node_test(tree.child(2), Some("D"),  0.0,51.0,  0.0);
    do_node_test(tree.child(3), Some("H"), 51.0,51.0,  0.0);
    do_node_test(tree.child(4), Some("A"),  0.0, 0.0, 51.0);
    do_node_test(tree.child(5), Some("E"), 51.0, 0.0, 51.0);
    do_node_test(tree.child(6), Some("B"),  0.0,51.0, 51.0);
    do_node_test(tree.child(7), Some("F"), 51.0,51.0, 51.0);
}
